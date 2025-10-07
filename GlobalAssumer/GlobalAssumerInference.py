#!/usr/bin/env python
"""
Assumer Inference Script for Text-to-SQL Rule Generation

This script is designed to perform inference using a fine-tuned Assumer model
to generate new rules from condensed_rules.json dataset.
It replaces the original rules with the generated ones and outputs to a new file.

Features:
- Load fine-tuned model with LoRA adapters
- Process condensed_rules.json dataset
- Generate new rules using the model
- Output results with replaced rules to new JSON file
- Support for multi-GPU inference
- Configurable generation parameters

Example:
  python AssumerInference.py \
    --model /path/to/base/model \
    --adapter_path /path/to/fine_tuned_adapter \
    --input_file /path/to/condensed_rules.json \
    --output_file /path/to/new_rules.json \
    --db_root_path /path/to/databases \
    --num_samples 100 \
    --max_new_tokens 256

Author: ChatTB Team
Date: 2024
"""

from __future__ import annotations

# Standard library imports
import argparse
import json
import os
from typing import Dict, List, Optional, Iterable
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Progress bar
from tqdm import tqdm

# PyTorch and deep learning libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# PEFT (Parameter-Efficient Fine-Tuning) libraries
from peft import PeftModel

# Local imports for schema processing
from Process_model.SchemaInformation import SchemaInformation


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for inference configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with validation
    """
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Assumer model for rule generation"
    )
    
    # Model and data configuration
    parser.add_argument("--model", type=str, required=True, 
                       help="Base model directory path (e.g., /path/to/Qwen3-8B)")
    parser.add_argument("--adapter_path", type=str, required=True, 
                       help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to input condensed_rules.json file")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to output file with generated rules")
    parser.add_argument("--tableSchema_path", type=str, required=True, 
                       help="Path to table schema file")
    parser.add_argument("--instruction", type=str, 
                       default="Analyze the question and schema, output only the rules that apply.",
                       help="Instruction prompt for the model")
    parser.add_argument("--db_root_path", type=str, default="/home/ubuntu/walkiiiy/ChatTB/Database_train", 
                       help="Root directory containing database files (<db_id>/<db_id>.sqlite)")
    parser.add_argument("--schema_rows", type=int, default=0, 
                       help="Number of sample rows to include per table in schema (0 to disable)")
    parser.add_argument("--trust_remote_code", action="store_true", 
                       help="Allow execution of custom modeling code from model hub")
    
    # Inference parameters
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of samples to process (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation (0.0 for deterministic)")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty for generation")
    
    # Performance settings
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--use_qlora", action="store_true",
                       help="Use QLoRA quantization for memory efficiency")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--tf32", action="store_true",
                       help="Use TensorFloat-32 precision on Ampere GPUs")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of worker threads for I/O operations")
    parser.add_argument("--save_frequency", type=int, default=100,
                       help="Save intermediate results every N samples")
    
    # Logging
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save intermediate results during processing")
    
    # Resume functionality
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output file if it exists")
    parser.add_argument("--resume_from_file", type=str, default=None,
                       help="Specific file to resume from (overrides --output_file)")

    args = parser.parse_args()
        
    # Validate input paths
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"Model path not found or not a directory: {args.model}")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found or not a directory: {args.adapter_path}")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    return args


def build_bnb_config(use_qlora: bool) -> Optional[BitsAndBytesConfig]:
    """
    Build BitsAndBytes configuration for QLoRA quantization.
    
    Args:
        use_qlora (bool): Whether to enable QLoRA quantization
        
    Returns:
        Optional[BitsAndBytesConfig]: Quantization config if use_qlora=True, None otherwise
    """
    if not use_qlora:
        return None
        
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    """
    Load and configure the tokenizer for the model.
    
    Args:
        model_name (str): Path to the model directory
        trust_remote_code (bool): Whether to trust remote code from model hub
        
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        trust_remote_code=trust_remote_code
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "right"
    
    return tokenizer


def load_model_with_adapter(model_name: str, adapter_path: str, bnb_config: Optional[BitsAndBytesConfig], 
                           bf16: bool, tf32: bool, trust_remote_code: bool) -> AutoModelForCausalLM:
    """
    Load the base model and apply LoRA adapter.
    
    Args:
        model_name (str): Path to the base model directory
        adapter_path (str): Path to the LoRA adapter
        bnb_config (Optional[BitsAndBytesConfig]): Quantization configuration
        bf16 (bool): Whether to use bfloat16 precision
        tf32 (bool): Whether to use TensorFloat-32 precision
        trust_remote_code (bool): Whether to trust remote code from model hub
        
    Returns:
        AutoModelForCausalLM: Model with LoRA adapter applied
    """
    # Enable TensorFloat-32 if requested
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    
    # Set dtype for model loading
    dtype = torch.bfloat16 if bf16 else None
    
    print(f"📖 Loading base model from: {model_name}")
    print(f"   - Quantization: {'QLoRA' if bnb_config else 'None'}")
    print(f"   - Precision: {'bfloat16' if bf16 else 'default'}")
    print(f"   - TF32: {tf32}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    
    print(f"🔧 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded successfully!")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Device: {next(model.parameters()).device}")
    
    return model


def build_inference_prompt(instruction: str, schema: str, question: str) -> str:
    """
    Build inference prompt from instruction, schema, and question.
    
    Args:
        instruction (str): Task instruction for the model
        schema (str): Database schema information
        question (str): Natural language question
        
    Returns:
        str: Formatted inference prompt
    """
    instruction = '''
    You are an expert in analyzing database schemas and user questions to infer possible rules.  
Rules describe **special mappings or operations** that must be followed when interpreting the question and generating SQL.  

The output format must always be:

rules:
[condition]: [operation], 
[condition]: [operation],
...

Rules should be concise, accurate, and schema-faithful. You have to make sure all the table and column names belongs to the schema.
### Examples:

rules:
When answering about "heads of the departments": use table "head" instead of "departments" for counting heads.

When the question asks for customer information: use table "Customers" instead of "customers" with exact case and quotes. If the question involves multiple tables, join "Customers_cards" as T1 with "Customers" as T2 on T1.customer_id = T2.customer_id using an inner match. If the question refers to a table named "customer" instead of "customers", use the correct table name with exact case and quotes. 

When the question asks for students who have taken courses: join table "student" with "enrollment" on student.id = enrollment.student_id, and join with "course" on course.id = enrollment.course_id.

### Now process the following:
    '''
    
    # Build the complete prompt
    prompt = (
        f"Instruction:{instruction}\n\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"generated rules:\n"
    )
    
    return prompt


def iter_rules_items(rules_json_path: str) -> Iterable[Dict]:
    """
    Iterate over items in a rules JSON file.
    
    Args:
        rules_json_path (str): Path to the rules JSON file
        
    Yields:
        Dict: Individual rule items from the file
    """
    with open(rules_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Handle both list and dict formats
    if isinstance(data, list):
        for item in data:
            yield item
    elif isinstance(data, dict):
        for _k, item in data.items():
            yield item
    else:
        raise ValueError("Unsupported rules file structure. Expect list or dict.")


def generate_rules_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                        prompts: List[str], generation_config: GenerationConfig, 
                        verbose: bool = False) -> List[str]:
    """
    Generate rules using the model in batch mode for better performance.
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer for the model
        prompts: List of input prompts
        generation_config: Generation configuration
        verbose: Whether to enable verbose logging
        
    Returns:
        List[str]: List of generated rules text
    """
    if not prompts:
        return []
    
    # Tokenize all inputs at once
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    if verbose:
        print(f"Batch size: {len(prompts)}, Input shape: {inputs['input_ids'].shape}")
    
    # Generate for all prompts at once
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens for each sample
    generated_texts = []
    input_lengths = inputs['attention_mask'].sum(dim=1)
    
    for i, output in enumerate(outputs):
        input_length = input_lengths[i].item()
        generated_tokens = output[input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text.strip())
    
    if verbose:
        print(f"Generated {len(generated_texts)} responses")
    
    return generated_texts


def generate_rules(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                  prompt: str, generation_config: GenerationConfig, 
                  verbose: bool = False) -> str:
    """
    Generate rules using the model (single prompt wrapper for backward compatibility).
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        generation_config: Generation configuration
        verbose: Whether to enable verbose logging
        
    Returns:
        str: Generated rules text
    """
    results = generate_rules_batch(model, tokenizer, [prompt], generation_config, verbose)
    return results[0] if results else ""


def extract_rules_from_generated_text(generated_text: str) -> List[str]:
    """
    Extract rules from generated text.
    
    Args:
        generated_text (str): Generated text from the model
        
    Returns:
        List[str]: List of extracted rules
    """
    rules = []
    
    # Look for rules section
    if "rules:" in generated_text.lower():
        # Extract content after "rules:"
        rules_section = generated_text.split("rules:")[-1].strip()
        
        # Split by lines and clean up
        for line in rules_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Remove leading numbers or bullets
                if ':' in line:
                    rules.append(line)
    else:
        # If no clear rules section, try to extract meaningful lines
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and len(line) > 10 and ':' in line:
                rules.append(line)
    
    return rules


def load_existing_results(output_file: str) -> Dict:
    """
    Load existing results from output file for resume functionality.
    
    Args:
        output_file (str): Path to existing output file
        
    Returns:
        Dict: Existing results or empty dict if file doesn't exist
    """
    if not os.path.exists(output_file):
        return {}
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"📂 Loaded existing results from: {output_file}")
        print(f"   Found {len(results)} existing samples")
        return results
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results from {output_file}: {e}")
        return {}


def get_next_sample_index(existing_results: Dict, total_samples: int) -> int:
    """
    Get the next sample index to process based on existing results.
    
    Args:
        existing_results (Dict): Existing results
        total_samples (int): Total number of samples in dataset
        
    Returns:
        int: Next sample index to process
    """
    if not existing_results:
        return 0
    
    # Find the highest existing index
    existing_indices = [int(k) for k in existing_results.keys() if k.isdigit()]
    if not existing_indices:
        return 0
    
    max_index = max(existing_indices)
    next_index = max_index + 1
    
    # Check if we've processed all samples
    if next_index >= total_samples:
        print("✅ All samples have already been processed!")
        return -1
    
    return next_index


class AsyncFileSaver:
    """Asynchronous file saver to avoid blocking the main processing loop."""
    
    def __init__(self, output_file: str, save_frequency: int = 100):
        self.output_file = output_file
        self.save_frequency = save_frequency
        self.results_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        self.last_save_count = 0
        self.save_count = 0
        
    def _save_worker(self):
        """Worker thread that saves results to file."""
        while True:
            try:
                results = self.results_queue.get(timeout=1.0)
                if results is None:  # Shutdown signal
                    break
                    
                # Create backup filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{self.output_file}.backup_{timestamp}"
                
                # Save to main file
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                # Also save backup
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                self.save_count += 1
                print(f"💾 Saved intermediate results (#{self.save_count}) to: {self.output_file}")
                print(f"   Backup saved to: {backup_file}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Error saving file: {e}")
    
    def save_if_needed(self, results: Dict, processed_count: int):
        """Save results if it's time to do so."""
        if processed_count - self.last_save_count >= self.save_frequency:
            self.results_queue.put(results.copy())
            self.last_save_count = processed_count
    
    def final_save(self, results: Dict):
        """Save final results and shutdown."""
        self.results_queue.put(results)
        self.results_queue.put(None)  # Shutdown signal
        self.save_thread.join()
        print(f"✅ Final save completed. Total saves: {self.save_count + 1}")


def load_schema_cache(tableSchema_path: str) -> Dict[str, str]:
    """
    Load and cache all schema information to avoid repeated file I/O.
    
    Args:
        tableSchema_path (str): Path to the table schema file
        
    Returns:
        Dict[str, str]: Dictionary mapping db_id to schema string
    """
    print(f"Loading schema cache from: {tableSchema_path}")
    with open(tableSchema_path, 'r', encoding='utf-8') as f:
        tableSchema = json.load(f)
    print(f"Loaded {len(tableSchema)} schemas into cache")
    return tableSchema


def process_dataset_batch(tableSchema: Dict[str, str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                         batch_items: List[Dict], instruction: str, generation_config: GenerationConfig,
                         verbose: bool = False, batch_idx: int = 0) -> List[Dict]:
    """
    Process a batch of items for inference.
    
    Args:
        tableSchema: Cached schema information
        model: The fine-tuned model
        tokenizer: Tokenizer for the model
        batch_items: List of items to process
        instruction: Instruction prompt
        generation_config: Generation configuration
        verbose: Whether to enable verbose logging
        batch_idx: Batch index for logging
        
    Returns:
        List[Dict]: List of processed results
    """
    if not batch_items:
        return []
    
    if verbose:
        print(f"🔄 Processing batch {batch_idx} with {len(batch_items)} items")
    
    # Prepare prompts for batch processing
    prompts = []
    valid_items = []
    error_count = 0
    
    for item in batch_items:
        question = item.get("question", "").strip()
        db_id = item.get("db_id", "").strip()
        original_rules = item.get("rules", []) or []
        
        # Get schema from cache
        schema = tableSchema.get(db_id, "")
        if not schema:
            if verbose:
                print(f"⚠️  Warning: No schema found for db_id: {db_id}")
            # Create error result
            result_item = item.copy()
            result_item["original_rules"] = original_rules
            result_item["rules"] = ["failed to find schema"]
            valid_items.append(result_item)
            error_count += 1
            continue
        
        # Build inference prompt
        prompt = build_inference_prompt(instruction, schema, question)
        prompts.append(prompt)
        valid_items.append(item)
    
    if not prompts:
        if verbose:
            print(f"❌ Batch {batch_idx}: No valid prompts to process")
        return valid_items
    
    if verbose:
        print(f"📝 Batch {batch_idx}: Processing {len(prompts)} prompts, {error_count} errors")
    
    # Generate rules in batch
    try:
        batch_start_time = time.time()
        generated_texts = generate_rules_batch(model, tokenizer, prompts, generation_config, verbose)
        batch_time = time.time() - batch_start_time
        
        if verbose:
            print(f"⚡ Batch {batch_idx}: Generated {len(generated_texts)} responses in {batch_time:.2f}s")
        
        # Process results
        results = []
        for i, (item, generated_text) in enumerate(zip(valid_items, generated_texts)):
            original_rules = item.get("rules", []) or []
            new_rules = extract_rules_from_generated_text(generated_text)
            
            result_item = item.copy()
            result_item["original_rules"] = original_rules
            result_item["rules"] = new_rules
            
            if verbose and i < 2:  # Show first 2 examples
                print(f"📋 Sample {i+1}: Generated {len(new_rules)} rules")
                if new_rules:
                    print(f"   First rule: {new_rules[0][:100]}...")
            
            results.append(result_item)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch {batch_idx} processing: {e}")
        # Return items with error rules
        results = []
        for item in valid_items:
            original_rules = item.get("rules", []) or []
            result_item = item.copy()
            result_item["original_rules"] = original_rules
            result_item["rules"] = ["batch processing error"]
            results.append(result_item)
        return results


def process_dataset(tableSchema_path: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                   input_file: str, output_file: str, instruction: str, 
                   db_root_path: str, schema_rows: int, num_samples: int,
                   generation_config: GenerationConfig, batch_size: int,
                   verbose: bool, save_intermediate: bool, resume: bool = False,
                   save_frequency: int = 100) -> None:
    """
    Process the entire dataset and generate new rules using optimized batch processing.
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer for the model
        input_file: Path to input condensed_rules.json
        output_file: Path to output file
        instruction: Instruction prompt
        db_root_path: Root directory for database files
        schema_rows: Number of sample rows to include
        num_samples: Number of samples to process (-1 for all)
        generation_config: Generation configuration
        batch_size: Batch size for processing
        verbose: Whether to enable verbose logging
        save_intermediate: Whether to save intermediate results
        resume: Whether to resume from existing output file
    """
    # Load schema cache once
    tableSchema = load_schema_cache(tableSchema_path)
    
    # Initialize async file saver
    file_saver = AsyncFileSaver(output_file, save_frequency) if save_intermediate else None
    
    # Load existing results if resuming
    if resume:
        results = load_existing_results(output_file)
        if results:
            print(f"🔄 Resuming from existing results with {len(results)} samples")
    else:
        results = {}
    
    processed_count = 0
    total_count = 0
    
    print(f"Loading dataset from: {input_file}")
    
    # Count total samples
    for _ in iter_rules_items(input_file):
        total_count += 1
    
    print(f"Total samples in dataset: {total_count}")
    if num_samples > 0:
        print(f"Processing first {num_samples} samples")
    else:
        print("Processing all samples")
    
    # Determine starting point for processing
    if resume and results:
        start_index = get_next_sample_index(results, total_count)
        if start_index == -1:
            print("✅ All samples have already been processed!")
            return
        print(f"🔄 Resuming from sample index: {start_index}")
    else:
        start_index = 0
    
    start_time = time.time()
    
    # Calculate total samples to process
    total_to_process = min(total_count - start_index, num_samples) if num_samples > 0 else total_count - start_index
    
    # Create progress bar
    pbar = tqdm(
        total=total_to_process,
        desc="Processing samples",
        unit="samples",
        initial=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process in batches
    current_index = 0
    batch_items = []
    batch_idx = 0
    
    for item in iter_rules_items(input_file):
        # Skip samples that have already been processed
        if current_index < start_index:
            current_index += 1
            continue
            
        if num_samples > 0 and processed_count >= num_samples:
            break
        
        batch_items.append(item)
        
        # Process batch when it reaches the desired size
        if len(batch_items) >= batch_size:
            batch_results = process_dataset_batch(
                tableSchema, model, tokenizer, batch_items, instruction, 
                generation_config, verbose, batch_idx
            )
            
            # Add results to main results dict
            for i, result_item in enumerate(batch_results):
                results[str(current_index - len(batch_items) + i + 1)] = result_item
            
            processed_count += len(batch_items)
            current_index += len(batch_items)
            batch_items = []
            batch_idx += 1
            
            # Update progress bar
            pbar.update(len(batch_results))
            
            # Update progress bar description with stats
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            remaining = (total_to_process - processed_count) * avg_time
            pbar.set_description(f"Processing samples (avg: {avg_time:.2f}s/sample, ETA: {remaining/60:.1f}min)")
            
            # Save intermediate results asynchronously
            if file_saver:
                file_saver.save_if_needed(results, processed_count)
        else:
            current_index += 1
    
    # Process remaining items in the last batch
    if batch_items:
        batch_results = process_dataset_batch(
            tableSchema, model, tokenizer, batch_items, instruction, 
            generation_config, verbose, batch_idx
        )
        
        # Add results to main results dict
        for i, result_item in enumerate(batch_results):
            results[str(current_index - len(batch_items) + i + 1)] = result_item
        
        processed_count += len(batch_items)
        pbar.update(len(batch_results))
    
    # Close progress bar
    pbar.close()
    
    # Save final results
    print(f"\n💾 Saving final results to: {output_file}")
    if file_saver:
        file_saver.final_save(results)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    total_time = time.time() - start_time
    print(f"\n🎉 Inference completed!")
    print(f"📊 Statistics:")
    print(f"   - Processed: {processed_count} samples")
    print(f"   - Total time: {total_time/60:.2f} minutes")
    print(f"   - Average time per sample: {total_time/processed_count:.2f} seconds")
    print(f"   - Throughput: {processed_count/(total_time/60):.1f} samples/minute")
    print(f"   - Results saved to: {output_file}")


def main() -> None:
    """
    Main function to orchestrate the inference process.
    """
    print("🚀 Starting Assumer Inference Process")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_args()
    print(f"📁 Input file: {args.input_file}")
    print(f"📁 Output file: {args.output_file}")
    print(f"🤖 Base model: {args.model}")
    print(f"🔧 Adapter path: {args.adapter_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load tokenizer and model
    print("\n📖 Loading tokenizer and model...")
    tokenizer = load_tokenizer(args.model, args.trust_remote_code)
    bnb_config = build_bnb_config(args.use_qlora)
    model = load_model_with_adapter(args.model, args.adapter_path, bnb_config, 
                                   args.bf16, args.tf32, args.trust_remote_code)
    print("✅ Model and tokenizer loaded successfully!")
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print(f"\n⚙️  Generation configuration:")
    print(f"   - Max new tokens: {args.max_new_tokens}")
    print(f"   - Temperature: {args.temperature}")
    print(f"   - Do sample: {args.do_sample}")
    print(f"   - Top-p: {args.top_p}")
    print(f"   - Top-k: {args.top_k}")
    
    # Determine output file for resume functionality
    output_file = args.output_file
    if args.resume_from_file:
        output_file = args.resume_from_file
        print(f"🔄 Using resume file: {output_file}")
    elif args.resume and os.path.exists(args.output_file):
        print(f"🔄 Resuming from existing file: {args.output_file}")
    
    # Process dataset
    print(f"\n🔄 Starting inference...")
    process_dataset(
        tableSchema_path=args.tableSchema_path,
        model=model,
        tokenizer=tokenizer,
        input_file=args.input_file,
        output_file=output_file,
        instruction=args.instruction,
        db_root_path=args.db_root_path,
        schema_rows=args.schema_rows,
        num_samples=args.num_samples,
        generation_config=generation_config,
        batch_size=args.batch_size,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        resume=args.resume or args.resume_from_file is not None,
        save_frequency=args.save_frequency,
    )
    
    print("🎉 Inference completed successfully!")


if __name__ == "__main__":
    main()
