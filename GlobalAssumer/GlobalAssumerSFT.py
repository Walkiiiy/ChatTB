#!/usr/bin/env python
"""
Fine-tune local Hugging Face causal LMs with LoRA/QLoRA using TRL's SFTTrainer.

This script is designed to fine-tune Qwen3 models on custom datasets for Text-to-SQL tasks.
It supports LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) for efficient training.

Features:
- Load dataset from condensed_rules.json files containing Text-to-SQL examples
- Generate training prompts from unified rules (no longer distinguishing between definitional and operational)
- Optional chat template application if tokenizer supports it
- LoRA or QLoRA (NF4/FP4) with bitsandbytes for memory efficiency
- Gradient checkpointing, packing, FSDP/DDP via accelerate config
- Saves adapter weights as safetensors
- Real-time logging of prompts and training loss
- Resume training from checkpoints or fine-tuned models

Example:
  # Start new training
  python AssumerSFT.py \
    --model /path/to/local/model \
    --rules_file /path/to/condensed_rules.json \
    --output_dir /path/to/outputs \
    --lora --max_steps 1000 --per_device_train_batch_size 2
  
  # Resume from a checkpoint
  python AssumerSFT.py \
    --model /path/to/local/model \
    --rules_file /path/to/condensed_rules.json \
    --output_dir /path/to/new_outputs \
    --resume_from_checkpoint /path/to/outputs/checkpoint-500 \
    --lora --max_steps 1000
  
  # Continue training from a fine-tuned model
  python AssumerSFT.py \
    --model /path/to/local/model \
    --rules_file /path/to/condensed_rules.json \
    --output_dir /path/to/new_outputs \
    --resume_from_model /path/to/fine_tuned_model \
    --lora --max_steps 1000

Author: ChatTB Team
Date: 2024
"""

from __future__ import annotations

# Standard library imports
import argparse
import json
import os
from typing import Dict, List, Optional, Iterable

# PyTorch and deep learning libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# PEFT (Parameter-Efficient Fine-Tuning) libraries
from peft import LoraConfig

# TRL (Transformer Reinforcement Learning) library for SFT
from trl import SFTTrainer, SFTConfig

# Local imports for schema processing
from Process_model.SchemaInformation import SchemaInformation


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for fine-tuning configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with validation
        
    Raises:
        FileNotFoundError: If model or rules file paths are invalid
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 models with LoRA/QLoRA for Text-to-SQL tasks using condensed rules"
    )
    
    # Model and data configuration
    parser.add_argument("--model", type=str, required=True, 
                       help="Local model directory path (e.g., /path/to/Qwen3-8B)")
    parser.add_argument("--rules_file", type=str, required=True, 
                       help="Path to condensed_rules.json file containing Text-to-SQL examples")
    parser.add_argument("--instruction", type=str, 
                       default="Analyze the question and schema, output only the rules that apply.",
                       help="Instruction prompt for the model")
    parser.add_argument("--db_root_path", type=str, default="/home/ubuntu/walkiiiy/ChatTB/Database_train", 
                       help="Root directory containing database files (<db_id>/<db_id>.sqlite)")
    parser.add_argument("--schema_rows", type=int, default=0, 
                       help="Number of sample rows to include per table in schema (0 to disable)")
    parser.add_argument("--skip_no_rules", action="store_true", 
                       help="Skip training samples that don't have rules")
    parser.add_argument("--use_chat_template", action="store_true", 
                       help="Apply model's chat template to messages (for chat-style prompts)")
    parser.add_argument("--trust_remote_code", action="store_true", 
                       help="Allow execution of custom modeling code from model hub")

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the fine-tuned model and checkpoints")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                       help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum number of training steps (-1 for epoch-based training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Batch size per device (reduce if OOM)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of steps to accumulate gradients before optimizer step")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay for regularization")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Ratio of total training steps for warmup")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--max_prompt_length", type=int, default=None,
                       help="Maximum sequence length for training")
    parser.add_argument("--packing", action="store_true", 
                       help="Enable sample packing for training efficiency")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 mixed precision training")
    parser.add_argument("--tf32", action="store_true",
                       help="Use TensorFloat-32 precision on Ampere GPUs")

    # LoRA/QLoRA configuration
    parser.add_argument("--lora", action="store_true",
                       help="Enable LoRA (Low-Rank Adaptation) fine-tuning")
    parser.add_argument("--qlora", action="store_true",
                       help="Enable QLoRA (Quantized LoRA) fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank (higher = more parameters, lower = more efficient)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, 
                       default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                       help="Comma-separated list of module names to apply LoRA to")

    # Logging and checkpointing
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log training metrics every N steps")
    parser.add_argument("--report_to", type=str, default="none",
                       help="Logging backend (none, wandb, tensorboard)")
    
    # Resume training functionality
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume training from (e.g., /path/to/checkpoint-1000)")
    parser.add_argument("--resume_from_model", type=str, default=None,
                       help="Path to fine-tuned model directory to continue training from")

    args = parser.parse_args()

    # Default to LoRA if neither LoRA nor QLoRA is specified
    if not args.lora and not args.qlora:
        args.lora = True
        
    # Validate resume arguments
    if args.resume_from_checkpoint and args.resume_from_model:
        raise ValueError("Cannot specify both --resume_from_checkpoint and --resume_from_model. Choose one.")
    
    if args.resume_from_checkpoint and not os.path.isdir(args.resume_from_checkpoint):
        raise FileNotFoundError(f"Resume checkpoint path not found or not a directory: {args.resume_from_checkpoint}")
    
    if args.resume_from_model and not os.path.isdir(args.resume_from_model):
        raise FileNotFoundError(f"Resume model path not found or not a directory: {args.resume_from_model}")
        
    # Validate input paths
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"Model path not found or not a directory: {args.model}")
    if not os.path.exists(args.rules_file):
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")
        
    return args


def build_bnb_config(qlora: bool) -> Optional[BitsAndBytesConfig]:
    """
    Build BitsAndBytes configuration for QLoRA quantization.
    
    Args:
        qlora (bool): Whether to enable QLoRA quantization
        
    Returns:
        Optional[BitsAndBytesConfig]: Quantization config if qlora=True, None otherwise
        
    Note:
        QLoRA uses 4-bit quantization with NF4 data type for memory efficiency.
        Double quantization further reduces memory usage by quantizing quantization constants.
    """
    if not qlora:
        return None
        
    return BitsAndBytesConfig(
        load_in_4bit=True,                    # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",           # Use NF4 quantization (better than int4)
        bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype for efficiency
        bnb_4bit_use_double_quant=True,      # Use double quantization for memory savings
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    """
    Load and configure the tokenizer for the model.
    
    Args:
        model_name (str): Path to the model directory
        trust_remote_code (bool): Whether to trust remote code from model hub
        
    Returns:
        AutoTokenizer: Configured tokenizer with proper padding settings
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        trust_remote_code=trust_remote_code
    )
    
    # Ensure pad token is set for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set padding side for causal language modeling
    tokenizer.padding_side = "right"
    
    return tokenizer


def load_model(model_name: str, bnb_config: Optional[BitsAndBytesConfig], bf16: bool, tf32: bool, trust_remote_code: bool) -> AutoModelForCausalLM:
    """
    Load the causal language model with optional quantization and precision settings.
    
    Args:
        model_name (str): Path to the model directory
        bnb_config (Optional[BitsAndBytesConfig]): Quantization configuration for QLoRA
        bf16 (bool): Whether to use bfloat16 precision
        tf32 (bool): Whether to use TensorFloat-32 precision
        trust_remote_code (bool): Whether to trust remote code from model hub
        
    Returns:
        AutoModelForCausalLM: Loaded model with specified configurations
        
    Note:
        - bf16: Reduces memory usage while maintaining training stability
        - tf32: Improves performance on Ampere GPUs (RTX 30xx, A100, etc.)
        - device_map="auto": Automatically distributes model across available GPUs
    """
    # Enable TensorFloat-32 if requested (for Ampere GPUs)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    
    # Set dtype for model loading
    dtype = torch.bfloat16 if bf16 else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # Apply quantization if specified
        torch_dtype=dtype,               # Set precision
        device_map="auto",               # Auto-distribute across GPUs
        trust_remote_code=trust_remote_code,
    )
    
    # Clear any default generation config that might include sampling parameters
    # This prevents warnings when using deterministic generation
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        # Reset generation config to remove sampling parameters
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = False
    
    return model


def normalize_chat_example(example: dict, chat_field: str, tokenizer: AutoTokenizer, apply_template: bool) -> Dict[str, str]:
    """
    Normalize chat examples into text format for training.
    
    Args:
        example (dict): Example containing chat messages
        chat_field (str): Field name containing the chat messages
        tokenizer (AutoTokenizer): Tokenizer for applying chat templates
        apply_template (bool): Whether to apply the model's chat template
        
    Returns:
        Dict[str, str]: Normalized example with 'text' field
        
    Raises:
        ValueError: If chat_field doesn't contain a list of messages
    """
    messages = example.get(chat_field)
    if not isinstance(messages, list):
        raise ValueError("chat_json_field must contain a list of messages, e.g. [{role, content}, ...]")
        
    if apply_template:
        try:
            # Try to apply the model's chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,enable_thinking=False)
        except Exception:
            # Fallback to simple format if template fails
            text_lines = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
            text = "\n".join(text_lines)
    else:
        # Simple format without template
        text_lines = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        text = "\n".join(text_lines)
        
    return {"text": text}


# Removed is_definitional_rule function as we no longer distinguish between rule types


def build_io_pair(instruction: str, schema: str, question: str, rules: List[str]) -> str:
    """
    Build a training prompt from instruction, schema, question, and rules.
    
    Args:
        instruction (str): Task instruction for the model
        schema (str): Database schema information
        question (str): Natural language question
        rules (List[str]): List of rules (no longer distinguishing between types)
        
    Returns:
        str: Formatted training prompt with input and expected output
    """
    # Format the target rules with proper structure
    if rules:
        target = "\n\n".join([f"{rule}" for i, rule in enumerate(rules)])
    else:
        target = "No rules found."
    

    instruction ='''
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
    text = (
        f"Instruction:{instruction}\n\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"generated rules:\n{target}"
    )
    
    return text


def iter_rules_items(rules_json_path: str) -> Iterable[Dict]:
    """
    Iterate over items in a rules JSON file.
    
    Args:
        rules_json_path (str): Path to the rules JSON file
        
    Yields:
        Dict: Individual rule items from the file
        
    Raises:
        ValueError: If the file structure is not supported
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


def get_prompts_from_rules(max_prompt_length: int, rules_json_path: str, instruction: str, skip_no_rules: bool, db_root_path: str, schema_rows: int) -> List[Dict[str, str]]:
    """
    Generate training prompts from condensed rules file and database schemas.
    
    Args:
        rules_json_path (str): Path to condensed rules JSON file
        instruction (str): Instruction prompt for the model
        skip_no_rules (bool): Whether to skip samples without rules
        db_root_path (str): Root directory containing database files
        schema_rows (int): Number of sample rows to include per table
        
    Returns:
        List[Dict[str, str]]: List of training samples with formatted prompts
    """
    samples: List[Dict[str, str]] = []
    schema_helper = SchemaInformation()
    
    print(f"Loading rules from: {rules_json_path}")
    print(f"Database root path: {db_root_path}")
    
    schema_path = os.path.join(db_root_path,  "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_all = json.load(f)

    for item in iter_rules_items(rules_json_path):
        question = item.get("question", "").strip()
        db_id = item.get("db_id", "").strip()
        rules = item.get("rules", []) or []
        
        # Extract all rules (no longer distinguishing between types)
        # Rules are now simple strings in the condensed format
        rule_list = [rule.strip() for rule in rules if rule.strip()]
        
        # Skip samples without rules if requested
        if skip_no_rules and not rule_list:
            continue
        # Generate schema information if database exists
        # schema = ""
        # if db_id:
        #     db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        #     if os.path.exists(db_path):
        #         try:
        #             schema = schema_helper.generate_schema_info(
        #                 db_path, 
        #                 num_rows=(schema_rows if schema_rows > 0 else None)
        #             )
        #         except Exception as e:
        #             print(f"Warning: Failed to generate schema for {db_id}: {e}")
        #             raise e# make it immidiate error
        #             schema = ""
        if db_id in schema_all:
            schema = schema_all[db_id]
        else:
            print(f"Warning: Schema not found for {db_id}")
            schema = ""
        # Build the training prompt
        text = build_io_pair(instruction, schema, question, rule_list)
        if max_prompt_length is not None and len(text) > max_prompt_length:
            print(f"Warning: Text length {len(text)} exceeds max_prompt_length {max_prompt_length}")
            continue
        samples.append({"text": text})

    print(f"Generated {len(samples)} training samples")
    return samples


from datasets import Dataset
from dataclasses import dataclass
from typing import Any, Dict


def load_local_json_dataset(_: str):
    """
    Placeholder function for loading JSON datasets (disabled).
    
    Raises:
        NotImplementedError: Always raises since this functionality is disabled
    """
    raise NotImplementedError("Dataset loading disabled. Use --rules_file to generate prompts.")


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """
    Build LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        LoraConfig: Configuration for LoRA adaptation
    """
    # Parse target modules from comma-separated string
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    
    return LoraConfig(
        r=args.lora_r,                    # LoRA rank
        lora_alpha=args.lora_alpha,       # LoRA alpha scaling
        lora_dropout=args.lora_dropout,   # LoRA dropout rate
        bias="none",                      # Don't adapt bias terms
        task_type="CAUSAL_LM",            # Task type for causal language modeling
        target_modules=target_modules,    # Modules to apply LoRA to
    )


class CustomSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer with enhanced logging for prompts and loss.
    """
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add custom logging for prompts and loss.
        
        Args:
            logs (Dict[str, float]): Training metrics to log
            start_time (Optional[float]): Start time for logging (optional)
        """
        # Call parent logging method
        super().log(logs, start_time)
        
        # Extract and display loss if available
        if "train_loss" in logs:
            loss = logs["train_loss"]
            step = logs.get("step", "unknown")
            print(f"\n{'='*60}")
            print(f"Step {step} - Training Loss: {loss:.4f}")
            print(f"{'='*60}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to log sample prompts every 20 steps.
        
        Args:
            model: The model being trained
            inputs: Training inputs
            num_items_in_batch: Number of items in the batch
            
        Returns:
            Training loss
        """
        # Call parent training step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Log sample prompt every 20 steps (removed generation to save time)
        if hasattr(self, "state") and self.state.global_step % 20 == 0:
            self._log_sample_prompt(inputs)
            
        return loss
    
    def _log_sample_prompt(self, inputs):
        """
        Log a sample prompt to the console.
        
        Args:
            inputs: Training inputs containing text data
        """
        try:
            # Get a sample from the batch
            if "input_ids" in inputs:
                sample_ids = inputs["input_ids"][0]  # First sample in batch
                
                # Decode the prompt (truncate for readability)
                # Use processing_class (tokenizer) if available, fallback to tokenizer attribute
                tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                if tokenizer is not None:
                    sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
                    
                    # Truncate if too long
                    # if len(sample_text) > 500:
                        # sample_text = sample_text[:500] + "..."
                    
                    print(f"\n📝 Sample Prompt (Step {self.state.global_step}):")
                    print("-" * 40)
                    print(sample_text)
                    print("-" * 40)
                else:
                    print(f"\n📝 Sample Prompt IDs (Step {self.state.global_step}):")
                    print(f"Length: {len(sample_ids)} tokens")
                    print(f"First 10 tokens: {sample_ids[:10].tolist()}")
        except Exception as e:
            print(f"Warning: Failed to log sample prompt: {e}")



@dataclass
class CollatorMaskAfterDelimiter:
    """
    Simple collator that masks labels before a given text delimiter.
    """
    tokenizer: AutoTokenizer
    delimiter: str

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prefer already-tokenized features produced by SFTTrainer
        if "input_ids" in features[0]:
            input_ids_list = [
                (f["input_ids"].tolist() if torch.is_tensor(f["input_ids"]) else f["input_ids"]) for f in features
            ]
            attn_list = None
            if "attention_mask" in features[0]:
                attn_list = [
                    (f["attention_mask"].tolist() if torch.is_tensor(f["attention_mask"]) else f["attention_mask"]) for f in features
                ]
            pad_inputs: Dict[str, Any] = {"input_ids": input_ids_list}
            if attn_list is not None:
                pad_inputs["attention_mask"] = attn_list
            batch = self.tokenizer.pad(pad_inputs, padding=True, return_tensors="pt")
        else:
            # Fallback: raw text path
            texts = [f["text"] for f in features]
            batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Compute delimiter token ids
        delimiter_ids = self.tokenizer(self.delimiter, add_special_tokens=False)["input_ids"]
        if len(delimiter_ids) == 0:
            batch["labels"] = labels
            return batch

        # Mask labels before the response delimiter
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            start_index = -1
            for j in range(0, len(seq) - len(delimiter_ids) + 1):
                if seq[j:j + len(delimiter_ids)] == delimiter_ids:
                    start_index = j + len(delimiter_ids)
                    break
            if start_index == -1:
                labels[i, :] = -100
            else:
                labels[i, :start_index] = -100

        batch["labels"] = labels
        # Ensure attention_mask exists
        if "attention_mask" not in batch:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            batch["attention_mask"] = (batch["input_ids"] != pad_token_id).long()
        return batch


def main() -> None:
    """
    Main function to orchestrate the fine-tuning process.
    
    This function:
    1. Parses command line arguments
    2. Loads the model and tokenizer
    3. Generates training prompts from rules and schemas
    4. Sets up LoRA configuration
    5. Configures training parameters
    6. Initializes the custom trainer
    7. Runs the training loop
    8. Saves the fine-tuned model and configurations
    """
    print("🚀 Starting Qwen3 Fine-tuning Process")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_args()
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model path: {args.model}")
    print(f"📋 Rules file: {args.rules_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print("\n📖 Loading tokenizer and model...")
    
    # Determine the base model path for tokenizer
    if args.resume_from_model:
        print(f"🔄 Resuming from fine-tuned model: {args.resume_from_model}")
        tokenizer = load_tokenizer(args.resume_from_model, args.trust_remote_code)
    else:
        tokenizer = load_tokenizer(args.model, args.trust_remote_code)
    
    bnb_config = build_bnb_config(args.qlora)
    
    # Load model from appropriate source
    if args.resume_from_model:
        print(f"🔄 Loading model from fine-tuned checkpoint: {args.resume_from_model}")
        model = load_model(args.resume_from_model, bnb_config, args.bf16, args.tf32, args.trust_remote_code)
    else:
        model = load_model(args.model, bnb_config, args.bf16, args.tf32, args.trust_remote_code)
    
    print("✅ Model and tokenizer loaded successfully!")
    
    # Generate training prompts from rules and database schemas
    print("\n🔄 Generating training prompts...")
    prompt_records = get_prompts_from_rules(
        args.max_prompt_length,
        args.rules_file,
        args.instruction,
        args.skip_no_rules,
        args.db_root_path,
        args.schema_rows,
    )
    
    if len(prompt_records) == 0:
        raise ValueError("No prompts generated from rules file. Check --rules_file and contents.")
    
    # Convert to HuggingFace Dataset format
    ds = Dataset.from_list(prompt_records)
    print(f"✅ Dataset created with {len(ds)} samples")
    
    # Build LoRA configuration
    print("\n🔧 Setting up LoRA configuration...")
    lora_cfg = build_lora_config(args)
    print(f"   - LoRA rank: {args.lora_r}")
    print(f"   - LoRA alpha: {args.lora_alpha}")
    print(f"   - LoRA dropout: {args.lora_dropout}")
    print(f"   - Target modules: {args.target_modules}")
    
    # Configure training parameters
    print("\n⚙️  Configuring training parameters...")
    training_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        seed=args.seed,
        packing=args.packing,
        dataset_text_field="text",
        report_to=None if args.report_to == "none" else args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=None,
    )
    
    print(f"   - Batch size: {args.per_device_train_batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Max steps: {args.max_steps}")
    print(f"   - Mixed precision: {'bf16' if args.bf16 else 'fp32'}")
    
    # Initialize custom trainer with enhanced logging
    print("\n🎯 Initializing trainer...")
    response_template = "generated rules:\n"
    data_collator = CollatorMaskAfterDelimiter(tokenizer=tokenizer, delimiter=response_template)
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=training_cfg,
        data_collator=data_collator,
    )
    # Set tokenizer manually for prompt logging (using recommended approach)
    trainer.processing_class = tokenizer
    print("✅ Trainer initialized successfully!")
    
    # Handle resume from checkpoint if specified
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"🔄 Will resume training from checkpoint: {resume_from_checkpoint}")
    elif args.resume_from_model:
        # When resuming from a fine-tuned model, we don't need to specify a checkpoint
        # The model is already loaded with the fine-tuned weights
        print("🔄 Resuming from fine-tuned model weights (no checkpoint needed)")
    
    # Start training
    print("\n🏋️  Starting training...")
    print("📊 Training progress will be logged with sample prompts every 20 steps and loss values")
    print("⚡ Sample generation disabled for faster training")
    print("=" * 60)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the fine-tuned model
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training configuration for reference
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    print("✅ Fine-tuning completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")
    print("🎉 Training finished!")

if __name__ == "__main__":
    # Run the main script for command-line usage
    main()
    
    # Uncomment to run the example
    # example_usage()


