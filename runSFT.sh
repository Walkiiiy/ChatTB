#!/bin/bash
local_path=/home/ubuntu/walkiiiy/ChatTB
python $local_path/finetune_qwen3.py \
  --model $local_path/Process_model/models--Qwen3-8B \
  --rules_file $local_path/Bird_train/rules_res_type.json \
  --schema_rows 0 \
  --output_dir $local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider-Bird\
  --instruction "Based on the schema information of a dataset and a question, output only the definitional rules that apply to the quesiton and schema." \
  --lora --bf16 --max_steps -1 --num_train_epochs 3.0 --per_device_train_batch_size 1 --gradient_checkpointing
  # --resume_from_checkpoint $local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider/checkpoint-1000
  --resume_from_model $local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider \


  #一轮spider，三轮bird