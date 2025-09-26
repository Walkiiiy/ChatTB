python AssumerSFT.py \
 --model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
 --rules_file /home/ubuntu/walkiiiy/ChatTB/Bird_train/condensed_rules.json \
 --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Spider3_Bird3 \
 --num_train_epochs 2 \
 --qlora \
 --resume_from_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Spider3/checkpoint-3000