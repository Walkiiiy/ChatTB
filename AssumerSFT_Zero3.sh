#!/bin/bash

# DeepSpeed ZeRO-3 Fine-tuning Script for AssumerSFT_Zero3.py
# This script demonstrates how to use DeepSpeed ZeRO-3 for training large models

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Activate virtual environment
source .venv/bin/activate

# Set your paths here
MODEL_PATH="./Process_model/models--Qwen3-30B"  # e.g., /path/to/Qwen3-8B
RULES_FILE="./condensed_rules_all.json"  # e.g., ./condensed_rules_all.json
OUTPUT_DIR="./Process_model/models--Assumer_Mixed_Qwen30B"  # e.g., ./outputs/deepspeed_training

# DeepSpeed ZeRO-3 training with CPU offloading using custom config
deepspeed --num_gpus=2 AssumerSFT_Zero3.py \
    --model "$MODEL_PATH" \
    --deepspeed_config "./ds_config_zero3.json" \
    --rules_file "$RULES_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --qlora \
    --lora_r 2 \
    --lora_alpha 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --max_steps 1000 \
    --bf16 \
    --gradient_checkpointing \
    --max_seq_length 1024 \
    --save_steps 200 \
    --logging_steps 10 \
    --trust_remote_code

# Alternative: Use custom DeepSpeed config file
# deepspeed --num_gpus=2 AssumerSFT_Zero3.py \
#     --model "$MODEL_PATH" \
#     --rules_file "$RULES_FILE" \
#     --output_dir "$OUTPUT_DIR" \
#     # --db_root_path "$DB_ROOT_PATH" \
#     --deepspeed_config "./ds_config_zero3.json" \
#     --lora \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 2e-4 \
#     --max_steps 1000 \
#     --bf16 \
#     --gradient_checkpointing \
#     --save_steps 500 \
#     --logging_steps 10 \
#     --trust_remote_code

echo "DeepSpeed ZeRO-3 training completed!"
