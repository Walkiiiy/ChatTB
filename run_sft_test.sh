#!/bin/bash

# SFT Results Test Script
# This script tests the fine-tuned globalAssumer model by generating rules and then using DeepSeek to generate SQL

local_path=/home/ubuntu/walkiiiy/ChatTB

# Configuration
BASE_MODEL="$local_path/Process_model/models--Qwen3-8B"
ADAPTER_PATH="$local_path/Process_model/models--globalAssumer_Qwen3_8b_Spider-Bird"
DATASET="$local_path/Bird_dev/rules_res_type.json"
DB_ROOT_PATH="$local_path/Bird_dev/dev_databases"
OUTPUT_FILE="$local_path/sft_triple_comparison_results.json"
NUM_SAMPLES=100

echo "=== SFT Results Test Configuration ==="
echo "Base Model: $BASE_MODEL"
echo "Adapter Path: $ADAPTER_PATH"
echo "Dataset: $DATASET"
echo "Database Root: $DB_ROOT_PATH"
echo "Output File: $OUTPUT_FILE"
echo "Number of Samples: $NUM_SAMPLES"
echo ""

# Check if required files exist
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Error: Base model directory not found: $BASE_MODEL"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Error: Adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "❌ Error: Dataset file not found: $DATASET"
    exit 1
fi

if [ ! -d "$DB_ROOT_PATH" ]; then
    echo "❌ Error: Database root directory not found: $DB_ROOT_PATH"
    exit 1
fi

# Check for DeepSeek API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ Error: DEEPSEEK_API_KEY environment variable is not set."
    echo "Please set your DeepSeek API key:"
    echo "export DEEPSEEK_API_KEY='your_api_key_here'"
    exit 1
fi

echo "✅ All required files and configurations found"
echo ""

# Run the test
echo "🚀 Starting SFT Results Test..."
python $local_path/test_sft_results_api.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --dataset "$DATASET" \
  --db_root_path "$DB_ROOT_PATH" \
  --num_samples $NUM_SAMPLES \
  --output_file "$OUTPUT_FILE" \
  --max_new_tokens 256 \
  --temperature 0.1 \
  --do_sample \
  --deepseek_model "deepseek-coder" \
  --trust_remote_code

echo ""
echo "🎉 SFT Results Test completed!"
echo "📁 Results saved to: $OUTPUT_FILE"
