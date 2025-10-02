python AssumerSFTTest_Loss.py \
    --base_model /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B \
    --fine_tuned_models /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/checkpoint-13000 \
    --test_data /home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json \
    --output_dir /home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed/LossTest_13000_Bird_dev \
    --max_samples 100

# python AssumerSFTTest_Loss.py \
#   --base_model /path/to/base/model \
#   --fine_tuned_models /path/to/model1 /path/to/model2 /path/to/model3 \
#   --test_data /path/to/test_data.json \
#   --output_dir /path/to/results
