# Transformer Model for Genome Sequence Analysis
## BA-Thesis

Institut: Statistik, LMU Muenchen   
Project supervisors: Prof. Dr. Mina Rezaei, Hüseyin Anil Gündüz, Martin Binder   
Author: Noah Hurmer 

## DNABERT
This repository is forked from the original implementation of 'DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome' by

Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021;, btab083, https://doi.org/10.1093/bioinformatics/btab083

It was created to investigate the DNABERT model's proficiency on virus genome data and tasks in context to the *GenomeNet* project.   
Changes have been made to implement additional features as well as accomodate the new data and task setup.
For a more detailed setup description of DNABERT, please refer to [the original Readme](DNABERT_README.md).

### 1. Environment setup

A virtual Environment was created using miniconda.

```
conda create -n dnabert python=3.7
conda activate dnabert

conda install pytorch torchvision cudatoolkit -c pytorch

git clone https://github.com/minimops/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

Optionally, install apex for fp16 training.
change to a desired directory by `cd PATH_NAME`

```
git clone https://github.com/NVIDIA/apex
cd apex
```

For this to work under the current setup, it is necessary to remove the cuda version check lines in the apex install script.
The warning message will also suggest this. For the future, a more permanent solution would be to switch to amp.

```
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



### 2. Pre-train

#### 2.1 Data processing

Please see the template data at `/example/sample_data/pre`. If you are trying to pre-train DNABERT with your own data, please process you data into the same format as it. Note that the sequences are in kmer format, so you will need to convert your sequences into that. We also provide a custom function `seq2kmer`in `motif/motif_utils.py` for this conversion.



In the following example, we use DNABERT with kmer=6 as example.



#### 2.2 Model Training

```
cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=PATH_TO_DNABERT_REPO
export OUTPUT_PATH=output$KMER

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 24
    --add_run_info 'text'
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## 3. Fine-tune (Skip this section if you use fine-tuned model)

#### 3.1 Data processing


#### 3.2 Download pre-trained models (coming)

[virBERT]()

[virBERT-mask8]()

[virbert-stride3]()


#### 3.3 Fine-tune

```
cd examples

export KMER=6
export MODEL_PATH=PATH_TO_THE_PRETRAINED_MODEL
export DATA_PATH=sample_data/ft/$KMER
export OUTPUT_PATH=./ft/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 150 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
    --add_run_info 'text'
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## 4. Prediction

After the model is fine-tuned, we can get predictions by running

```$
export KMER=6
export MODEL_PATH=./ft/$KMER
export DATA_PATH=sample_data/ft/$KMER
export PREDICTION_PATH=./result/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 150 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 48
```

With the above command, the fine-tuned DNABERT model will be loaded from `MODEL_PATH` , and makes prediction on the `dev.tsv` file that saved in `DATA_PATH` and save the prediction result at `PREDICTION_PATH`.

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).