#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./data \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/8088/ \
  --load_path /home2/lzhang/base_model/ \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 12 \
  --test_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 2e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 52 \
  --seed 66 \
  --num_class 97 \
  | tee logs/8088.train.log 2>&1
