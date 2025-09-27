#!/bin/bash

# Example script for running SFT API SQL generation test with comprehensive reporting
# This script demonstrates how to use AssumerSFTTest_APIgenerateSQL.py with enhanced output
# Updated to support the new condensed_rules.json format

echo "🚀 Starting SFT API SQL Generation Test with Enhanced Reporting"
echo "=============================================================="

# Configuration
BASE_MODEL="./Process_model/models--Qwen3-8B"
ADAPTER_PATH="./Process_model/models--Assumer_Mixed/checkpoint-11000"
DATASET="./Spider_dev/condensed_rules.json"
DB_ROOT_PATH="./Spider_dev/database"
OUTPUT_DIR="./Process_model/models--Assumer_Mixed/SQL_test_results_11000_Spider_dev"
NUM_SAMPLES=100

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Configuration:"
echo "   Base Model: $BASE_MODEL"
echo "   Adapter Path: $ADAPTER_PATH"
echo "   Dataset: $DATASET (condensed_rules.json format)"
echo "   Database Root: $DB_ROOT_PATH"
echo "   Output Directory: $OUTPUT_DIR"
echo "   Number of Samples: $NUM_SAMPLES"
echo ""

# Check if required files exist
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Base model directory not found: $BASE_MODEL"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset file not found: $DATASET"
    exit 1
fi

if [ ! -d "$DB_ROOT_PATH" ]; then
    echo "❌ Database root directory not found: $DB_ROOT_PATH"
    exit 1
fi

# Check for DeepSeek API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  DEEPSEEK_API_KEY environment variable not set"
    echo "   Please set your DeepSeek API key:"
    echo "   export DEEPSEEK_API_KEY='your_api_key_here'"
    echo ""
    echo "   Or you can pass it directly with --deepseek_api_key parameter"
fi

# Run the test
echo "🔍 Running SFT API SQL generation test..."
python AssumerSFTTest_APIgenerateSQL.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --dataset "$DATASET" \
    --db_root_path "$DB_ROOT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens 256 \
    --temperature 0.1 \
    --deepseek_model "deepseek-coder"

echo ""
echo "✅ SFT API test completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📋 Generated files:"
echo "   - detailed_results.csv (detailed CSV with all sample results)"
echo "   - summary_statistics.csv (summary statistics table)"
echo "   - enhanced_test_results.json (comprehensive JSON report)"
echo "   - summary_report.txt (human-readable summary)"
echo "   - sft_test_results_analysis.png (main visualization plots)"
echo "   - detailed_sample_comparison.png (sample-by-sample heatmap)"
echo "   - sft_api_test.log (detailed test log with all logging output)"
echo ""
echo "🎯 Key Metrics to Review:"
echo "   - Accuracy comparison between SFT rules, dataset rules, and no rules"
echo "   - Improvement analysis showing effectiveness of different approaches"
echo "   - Error patterns and failure cases"
echo "   - Database-specific performance analysis"
