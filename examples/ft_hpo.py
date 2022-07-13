import argparse
from run_finetune import evaluate, MODEL_CLASSES, load_and_cache_examples, TOKEN_ID_GROUP
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
    get_linear_schedule_with_warmup,
)
import pandas as pd
from sql_db import create_connection
import numpy as np
import random
import re
from timeit import default_timer as timer
import torch
import joblib
from math import log2
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from os.path import exists
import os
import optuna
from optuna.trial import TrialState

DIRNUM = 0
N_TRAIN_EXAMPLES = .75


def create_tokenizer(tokenizer_class, args):
    return tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


def get_model(model_class, config, args):
    return model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


def create_config(config_class, args, num_labels):
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    if args.model_type in ["dnalong", "dnalongcat"]:
        assert args.max_seq_length % 512 == 0
    config.split = int(args.max_seq_length / args.max_tokens)
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden
    return config


def prepare_training(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = create_config(config_class, args, num_labels)

    tokenizer = create_tokenizer(tokenizer_class, args)

    model = get_model(model_class, config, args)

    # freeze layers
    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    return config, tokenizer, model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def objective(trial, args):
    # create trial arguments
    args.learning_rate = trial.suggest_float("learning_rate", 5e-6, 4e-4, log=True)
    args.per_gpu_train_batch_size = trial.suggest_int("per_gpu_train_batch_size", 5, int(log2(args.max_train_bs)))
    args.warmup_percent = trial.suggest_int("warmup_percent", 1, 4)
    # additional stuff
    # weight decay
    args.weight_decay = trial.suggest_int("weight_decay", -1, 3)
    # dropout probabilities
    args.hidden_dropout_prob = trial.suggest_int("hidden_dropout_prob", 1, 4)

    # map ints to percentage
    args.warmup_percent = args.warmup_percent * 0.05
    args.per_gpu_train_batch_size = 2 ** args.per_gpu_train_batch_size
    args.weight_decay = 10 ** -args.weight_decay
    args.hidden_dropout_prob = args.hidden_dropout_prob * 0.1

    # Setup CUDA, GPU & distributed training
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        # torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Create output directory if needed
    if args.local_rank in [-1, 0]:
        global DIRNUM
        output_dir = args.output_dir + "/run_" + str(args.gpu_id) + "_" + str(DIRNUM)
        #run_dir = "run" + str(DIRNUM)
        os.makedirs(output_dir, exist_ok=True)
        DIRNUM += 1

    with open(output_dir + "/opt_args.csv", "w") as f:
        f.write(", ".join(trial.params.keys()) + "\n")
        f.write(", ".join([str(x) for x in trial.params.values()]))

    # reproducability
    set_seed(args)

    # load model
    config, tokenizer, model = prepare_training(args)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

    # prepare training dataset
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # calculate training steps
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # train

    best_score = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )

    stop_count = 0
    rep_counter = 0
    t_start = timer()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print(len(epoch_iterator))
        for step, batch in enumerate(epoch_iterator):

            if step >= N_TRAIN_EXAMPLES * len(epoch_iterator):
                break

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # evaluate
                if global_step % int(args.logging_steps * 64 / args.train_batch_size) == 0:
                    results = evaluate(args, model, tokenizer, global_step, timestamp=timer() - t_start, prefix=output_dir)
                    print("\n\n", results["acc"], "\n")
                    # early stopping
                    if results["acc"] <= best_score:
                        stop_count += 1
                    else:
                        stop_count = 0
                        best_score = results["acc"]

                    # stop training when patience count is reached
                    # or it meets this stupid ill-defined metric shit
                    if stop_count == args.early_stop or results["acc"] == 0.5 and stop_count >= 1:
                        print("\nSTOPPING EARLY\n")
                        trial.report(results["acc"], rep_counter)
                        return best_score

                    if args.report_steps == -1 or global_step % int(
                            args.report_steps * 64 / args.train_batch_size) == 0:
                        print("REPORTING\n")
                        trial.report(results["acc"], rep_counter)
                        rep_counter += 1

                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            print("\nPRUNING TRIAL\n")
                            raise optuna.exceptions.TrialPruned()
        #
        # # write training file
        # logs = {}
        # for key, value in results.items():
        #     eval_key = "eval_{}".format(key)
        #     logs[eval_key] = value
        # loss_scalar = (tr_loss - logging_loss) / args.logging_steps
        # learning_rate_scalar = scheduler.get_lr()[0]
        # logs["learning_rate"] = learning_rate_scalar
        # logs["loss"] = loss_scalar
        # logging_loss = tr_loss

        # if not os.path.exists(args.output_dir + "/tr_args.csv"):
        #     headers = ",".join(["global_step", "learning_rate", "training_loss"]) + "\n"
        # else:
        #     headers = ""
        # with open(args.output_dir + "/tr_args.csv", "a") as writer:
        #     writer.write(headers + ",".join(
        #         str(x) for x in [global_step, logs.get("learning_rate"), logs.get("loss")]) + "\n")

    results = evaluate(args, model, tokenizer, global_step, timestamp=timer() - t_start, prefix=output_dir)
    if results["acc"] > best_score:
        best_score = results["acc"]
    print("REPORTING\n")
    trial.report(results["acc"], rep_counter)

    return best_score


if __name__ == "__main__":

    torch.set_printoptions(threshold=10_000)

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--study_name",
        default=None,
        type=str,
        required=True,
        help="Name of the optuna study",
    )
    parser.add_argument(
        "--n_process",
        default=2,
        type=int,
        help="number of processes used for data process",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument("--freeze", action="store_true", help="Freeze Bert layers except fo the classification one")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=150,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--max_train_bs", default=256, type=int, help="max training per gpu batch size for study",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--early_stop", default=0, type=int, help="set this to a positive integer if you want to perform early stop. The model will stop \
                                                    if the acc keep decreasing early_stop times",
    )

    parser.add_argument("--max_tokens", default=512, type=int, help="Number of tokens the model is limited to")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn_dropout", default=0.0, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float,
                        help="Linear warmup over warmup_percent*total_steps.")
    parser.add_argument("--do_ensemble_pred", action="store_true",
                        help="Whether to do ensemble prediction with kmer 3456.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--report_steps", type=int, default=-1, help="report to trial every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="To run on cluster with multiple gpus. Set this to the device id")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")

    args = parser.parse_args()

    location = args.output_dir + "/study.pkl"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize"
        , pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_warmup_steps=2,
                                                                          n_startup_trials=3),
                                              patience=2)

        , storage=create_connection("example.db"),
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=50, timeout=258500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    joblib.dump(study, location)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
