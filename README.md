## Requirements
* Python (tested on 3.9.7)
* CUDA (tested on 11.4)
* PyTorch (tested on 1.10.1)
* Transformers (tested on 4.15.0)
* numpy (tested on 1.20.3)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). 
```
Code
 |-- docred
 |    |-- train_annotated.json        
 |    |-- train_distant.json
 |    |-- dev.json
 |    |-- test.json
 |    |-- rel2id.json
 |    |-- ner2id.json
 |    |-- rel_info.json
 |-- logs
 |-- result
 |-- model
 |    |-- bert        
 |    |-- save_model
```
The core code of our inference module is contained in ` CCNet.py`.

## Training and Evaluation on DocRED
### At the first stage
Train the BERT-based model on DocRED with the following command:
```bash
>> sh scripts/run_bert.sh 
```
Hyper-parameters Setting
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./docred \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/ \
  --load_path ./model/save_model/ \
  --train_file train_annotated.json \
  -- train_base \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 20 \
  --test_batch_size 30 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 5e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 102 \
  --seed 66 \
  --num_class 97 \
  | tee logs/logs.train.log 2>&1
```
The trained base module is saved in `./model/save_model/`.

### At the two stage
First, you can set the base module to be loaded on line 389 of `train.py`.

Then, Train  our entire model on DocRED with the following command:
```bash
>> sh scripts/run_bert.sh  
```
Hyper-parameters Setting
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./docred \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/ \
  --load_path ./model/save_model/ \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 12 \
  --test_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 1e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 52 \
  --seed 66 \
  --num_class 97 \
  | tee logs/logs.train.log 2>&1
```
Note: The learning rate of the base module, `--encoder_lr`, needs to be dynamically set according to its performance.
Usually, with the better the performance of the base module, we set a smaller learning rate for it, and vice versa.

### Evaluating Model
First, you can set the model to be loaded on line 389 of `train.py`.

Then, Test our entire model on DocRED with the following command:

```bash
>> sh scripts/run_bert.sh 
```
Hyper-parameters Setting
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./docred \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/ \
  --load_path ./model/save_model/ \
  --train_file train_annotated.json \
  --test \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 12 \
  --test_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 1e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 52 \
  --seed 66 \
  --num_class 97 \
  | tee logs/logs.train.log 2>&1
```

The program will generate a test file `./result/result.json` in the official evaluation format. 
You can compress and submit it to Colab for the official test score.

