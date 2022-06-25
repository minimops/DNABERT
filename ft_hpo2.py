from src.transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='/content/results',    # output directory
    #evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    evaluation_strategy="steps",      # Evaluation is done (and logged) every eval_steps.
    eval_steps=1000,                  # Number of update steps between two evaluations
    per_device_eval_batch_size=64,    # batch size for evaluation
    save_total_limit=1,               # limit the total amount of checkpoints. Deletes the older checkpoints.
)

def model_init():
    model = BertForSequenceClassification.from_pretrained(model_name)
    for param in model.base_model.parameters():
        param.requires_grad = False
    return model

trainer = Trainer(
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics,     # metrics to be computed
    model_init=model_init                # Instantiate model before training starts
)

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 20),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }
def my_objective(metrics):
    return metrics["eval_loss"]

best_run = trainer.hyperparameter_search(direction="minimize", hp_space=my_hp_space, compute_objective=my_objective, n_trials=100)

with open("/content/drive/My Drive/cord19/best_run.json", "w+") as f:
  f.write(json.dumps(best_run.hyperparameters))