#!/bin/bash

python AssumerSFTTest_LocalModelSQL.py \
    --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
    --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/checkpoint-13000 \
    --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
    --dataset /home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json \
    --db_root_path /home/ubuntu/walkiiiy/ChatTB/Bird_dev/dev_databases \
    --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/SQL_test_results_13000_Bird_dev_ArcticSQL \
    --num_samples 100

