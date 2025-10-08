#!/bin/bash
# python AssumerSFTTest_LocalModelSQL.py \
#     --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
#     --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/checkpoint-13000 \
#     --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
#     --dataset /home/ubuntu/walkiiiy/ChatTB/Bird_dev/dev.json \
#     --db_root_path /home/ubuntu/walkiiiy/ChatTB/Bird_dev/dev_databases \
#     --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/SQL_test_results_13000_Bird_dev_ArcticSQL_Evidence \
#     --num_samples -1

# python GlobalAssumer/GlobalAssumerSFTTest_LocalModelSQL.py \
#     --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
#     --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_lossTweaked/checkpoint-11000 \
#     --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
#     --dataset /home/ubuntu/walkiiiy/ChatTB/Spider_dev/condensed_rules.json \
#     --db_root_path /home/ubuntu/walkiiiy/ChatTB/Spider_dev/database \
#     --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_lossTweaked/SQL_test_11000_Spider_dev_ArcticSQL \
#     --num_samples 100

python GlobalAssumer/GlobalAssumerSFTTest_LocalModelSQL.py \
    --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
    --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed10.7/checkpoint-1000 \
    --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
    --dataset /home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json \
    --db_root_path /home/ubuntu/walkiiiy/ChatTB/Bird_dev/database \
    --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed10.7/SQL_test_1000_Bird_dev_ArcticSQL \
    --num_samples 100
    
# python GlobalAssumer/GlobalAssumerSFTTest_LocalModelSQL_SFTonly.py \
#     --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
#     --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_Cleaned/checkpoint-10000 \
#     --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
#     --dataset /home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json \
#     --db_root_path /home/ubuntu/walkiiiy/ChatTB/Bird_dev/database \
#     --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_Cleaned/SQL_test_SFTonly_10000_Bird_dev_ArcticSQL \
#     --num_samples 100
# python AssumerSFTTest_LocalModelSQL.py \
#     --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
#     --adapter_path /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/checkpoint-13000 \
#     --sql_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B\
#     --dataset /home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json \
#     --db_root_path /home/ubuntu/walkiiiy/ChatTB/Bird_dev/dev_databases \
#     --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/SQL_test_results_13000_Bird_dev_ArcticSQL_Rules \
#     --num_samples -1