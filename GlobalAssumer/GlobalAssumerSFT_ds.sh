#! /bin/bash
deepspeed GlobalAssumer/GlobalAssumerSFT_ds.py \
    --model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
    --rules_json_path /home/ubuntu/walkiiiy/ChatTB/condensed_rules_all.json \
    --schema_path /home/ubuntu/walkiiiy/ChatTB/Database_train/schema.json \
    --batch_size 1 \
    --epochs 3 \
    --seed 42 \
    --device cuda \
    --deepspeed_config /home/ubuntu/walkiiiy/ChatTB/GlobalAssumer/GlobalAssumerSFT_ds_config.json

    