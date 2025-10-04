#python AssumerSFT.py \
# --model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
# --rules_file /home/ubuntu/walkiiiy/ChatTB/condensed_rules_all.json \
# --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed \
# --num_train_epochs 1 \
# --qlora \
#  --resume_from_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Spider3/checkpoint-3000

# 整合spider bird数据统一微调，Qwen 8b 3轮需要16h，
#  python AssumerSFT.py \
#   --model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
#   --rules_file /home/ubuntu/walkiiiy/ChatTB/condensed_rules_all.json \
#   --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed \
#   --num_train_epochs 3 \
#   --qlora \
#   --resume_from_checkpoint /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/checkpoint-11000
  # --resume_from_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Spider3/checkpoint-3000

 python GlobalAssumer/GlobalAssumerSFT_lossTweaked.py \
  --model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
  --rules_file /home/ubuntu/walkiiiy/ChatTB/condensed_rules_all.json \
  --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_lossTweaked \
  --num_train_epochs 3 \
  --tokenized_db_path /home/ubuntu/walkiiiy/ChatTB/Database_train/tokenizedDB_train.json \
  --schema_loss_weight 3.0 \
  --qlora \
  --skip_no_rules \
  --resume_from_checkpoint /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_lossTweaked/checkpoint-11000
 