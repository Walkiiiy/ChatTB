#!/bin/bash

# Assumer Inference Script
# This script runs inference using a fine-tuned Assumer model to generate new rules
# from condensed_rules.json dataset and output them to a new file.

# Configuration
MODEL_PATH="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
ADAPTER_PATH="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Assumer_Mixed_lossTweaked/checkpoint-13000"
INPUT_FILE="/home/ubuntu/walkiiiy/ChatTB/rules_shuffled.json"
OUTPUT_FILE="/home/ubuntu/walkiiiy/ChatTB/generated_rules.json"
DB_ROOT_PATH="/home/ubuntu/walkiiiy/ChatTB/Database_train"
TABLE_SCHEMA_PATH="/home/ubuntu/walkiiiy/ChatTB/Database_train/schema.json"

# Inference parameters
NUM_SAMPLES=-1  # Set to -1 to process all samples
MAX_NEW_TOKENS=65535
TEMPERATURE=0.1
DO_SAMPLE=false
BATCH_SIZE=8  # Reduced for better memory efficiency

# Performance settings
USE_QLORA=true
BF16=true
TF32=true
NUM_WORKERS=2
SAVE_FREQUENCY=50

# Logging
VERBOSE=true
SAVE_INTERMEDIATE=true

# Resume functionality
RESUME=false  # Set to true to enable resume from existing output file

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "🚀 Starting Assumer Inference"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Adapter: $ADAPTER_PATH"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Samples: $NUM_SAMPLES"
echo "Resume: $RESUME"
echo "================================"

# Check if output file exists for resume
if [ "$RESUME" = true ] && [ -f "$OUTPUT_FILE" ]; then
    echo "🔄 Found existing output file, will resume from it"
    echo "   Existing file: $OUTPUT_FILE"
else
    echo "🆕 Starting fresh inference"
fi

# Run inference
python GlobalAssumer/GlobalAssumerInference.py \
    --model "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --db_root_path "$DB_ROOT_PATH" \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --batch_size $BATCH_SIZE \
    --use_qlora \
    --bf16 \
    --tf32 \
    --trust_remote_code \
    --verbose \
    --save_intermediate \
    --tableSchema_path "$TABLE_SCHEMA_PATH" \
    --num_workers $NUM_WORKERS \
    --save_frequency $SAVE_FREQUENCY

echo "✅ Inference completed!"
echo "Results saved to: $OUTPUT_FILE"
