#!/bin/bash

# Test script for the fine-tuned model
local_path=/home/ubuntu/walkiiiy/ChatTB

python $local_path/test_finetuned_model.py \
  --base_model $local_path/Process_model/models--Qwen3-8B \
  --adapter_path $local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider-Bird\
  --rules_file $local_path/Spider_train/rules_res_type.json \
  --db_root_path $local_path/Spider_dev/database \
  --num_samples 3 \
  --max_new_tokens 256 \
  --temperature 0.1 \
  --do_sample
