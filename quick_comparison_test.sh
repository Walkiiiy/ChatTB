#!/bin/bash

# Quick Comparison Test Script - 快速对比测试版本
# 测试带有rules和不带rules的两种方式，对比性能差异

local_path=/home/ubuntu/walkiiiy/ChatTB

echo "🚀 Quick SFT Comparison Test (5 samples)"
echo "========================================="
echo "This test compares SQL generation with and without rules"
echo ""

# 检查API密钥
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ Error: DEEPSEEK_API_KEY environment variable is not set."
    echo "Please set your DeepSeek API key:"
    echo "export DEEPSEEK_API_KEY='your_api_key_here'"
    exit 1
fi

# 运行三重对比测试
python $local_path/test_sft_results_api.py \
  --base_model "$local_path/Process_model/models--Qwen3-8B" \
  --adapter_path "$local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider-Bird" \
  --dataset "$local_path/Bird_dev/rules_res_type.json" \
  --db_root_path "$local_path/Bird_dev/dev_databases" \
  --num_samples 5 \
  --output_file "$local_path/quick_triple_comparison_test_results.json" \
  --max_new_tokens 128 \
  --temperature 0.1 \
  --do_sample \
  --deepseek_model "deepseek-coder" \
  --trust_remote_code

echo ""
echo "✅ Quick triple comparison test completed!"
echo "📁 Results: $local_path/quick_triple_comparison_test_results.json"
echo ""
echo "📊 The results will show:"
echo "  - Accuracy with SFT-generated rules"
echo "  - Accuracy with dataset definitional rules (ground truth)"
echo "  - Accuracy without rules (baseline)"
echo "  - Performance comparisons between all three approaches"
