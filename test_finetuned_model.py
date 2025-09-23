#!/usr/bin/env python
"""
Test script for the fine-tuned Qwen3 model on Text-to-SQL rule extraction task.

This script loads the fine-tuned model and tests it with random samples from the dataset.
"""

import argparse
import json
import random
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from Process_model.SchemaInformation import SchemaInformation


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen3 model")
    
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model directory")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to the fine-tuned adapter directory")
    parser.add_argument("--rules_file", type=str, required=True,
                       help="Path to rules JSON file for testing")
    parser.add_argument("--db_root_path", type=str, default="./Spider_dev/database",
                       help="Root directory containing database files")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of random samples to test")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation (lower = more deterministic)")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during generation")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of custom modeling code")
    
    return parser.parse_args()


def load_model_and_tokenizer(base_model_path: str, adapter_path: str, trust_remote_code: bool):
    """Load the base model and fine-tuned adapter."""
    print(f"📖 Loading base model from: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        use_fast=True, 
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    
    # Load adapter
    print(f"🔧 Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("✅ Model and adapter loaded successfully!")
    return model, tokenizer


def load_test_samples(rules_file: str, db_root_path: str, num_samples: int) -> List[Dict]:
    """Load random samples from the rules file for testing."""
    print(f"📋 Loading test samples from: {rules_file}")
    
    with open(rules_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to list if it's a dict
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    
    # Filter items that have definitional rules
    valid_items = []
    for item in items:
        rules = item.get("rules", [])
        has_definitional = any(
            str(rule.get("type", "")).lower() in ["definitional", "definational"]
            for rule in rules
        )
        if has_definitional:
            valid_items.append(item)
    
    # Sample random items
    sampled_items = random.sample(valid_items, min(num_samples, len(valid_items)))
    
    print(f"✅ Selected {len(sampled_items)} test samples")
    return sampled_items


def extract_definitional_rules(rules: List[Dict]) -> List[str]:
    """Extract definitional rules from a rules list."""
    definitional_rules = []
    for rule in rules:
        if str(rule.get("type", "")).lower() in ["definitional", "definational"]:
            condition = rule.get("condition", "").strip()
            operation = rule.get("operation", "").strip()
            if condition and operation:
                definitional_rules.append(f"Condition: {condition}\nOperation: {operation}")
    return definitional_rules


def build_test_prompt(instruction: str, schema: str, question: str) -> str:
    """Build test prompt without the answer part."""
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"definitional rules:\n"
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int, 
                     temperature: float, do_sample: bool) -> str:
    """Generate response from the model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1 if do_sample else 1,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def test_model(model, tokenizer, test_samples: List[Dict], db_root_path: str, 
               instruction: str, max_new_tokens: int, temperature: float, do_sample: bool):
    """Test the model on the provided samples."""
    schema_helper = SchemaInformation()
    
    print(f"\n🧪 Testing model on {len(test_samples)} samples...")
    print("=" * 80)
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Test Sample {i}/{len(test_samples)}")
        print("-" * 60)
        
        # Extract sample information
        question = sample.get("question", "").strip()
        db_id = sample.get("db_id", "").strip()
        rules = sample.get("rules", [])
        
        # Get ground truth definitional rules
        ground_truth_rules = extract_definitional_rules(rules)
        
        # Generate schema
        schema = ""
        if db_id:
            db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    schema = schema_helper.generate_schema_info(db_path)
                except Exception as e:
                    print(f"Warning: Failed to generate schema for {db_id}: {e}")
        
        # Build test prompt
        prompt = build_test_prompt(instruction, schema, question)
        
        print(f"Question: {question}")
        print(f"Database: {db_id}")
        print(f"\nGround Truth Definitional Rules ({len(ground_truth_rules)}):")
        for j, rule in enumerate(ground_truth_rules, 1):
            print(f"  {j}. {rule}")
        
        print(f"\n🤖 Model Response:")
        print("-" * 40)
        
        # Generate model response
        response = generate_response(model, tokenizer, prompt, max_new_tokens, 
                                   temperature, do_sample)
        
        print(response)
        print("-" * 40)
        
        # Simple evaluation metrics
        print(f"\n📊 Evaluation:")
        print(f"  - Ground truth rules: {len(ground_truth_rules)}")
        print(f"  - Response length: {len(response)} characters")
        
        # Check if response contains some key elements
        has_condition = "condition:" in response.lower()
        has_operation = "operation:" in response.lower()
        print(f"  - Contains 'condition:': {has_condition}")
        print(f"  - Contains 'operation:': {has_operation}")
        
        print("\n" + "=" * 80)


def main():
    """Main function to run the test."""
    print("🚀 Starting Fine-tuned Model Test")
    print("=" * 60)
    
    args = parse_args()
    
    # Validate paths
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"Base model path not found: {args.base_model}")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if not os.path.exists(args.rules_file):
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, 
        args.adapter_path, 
        args.trust_remote_code
    )
    
    # Load test samples
    test_samples = load_test_samples(args.rules_file, args.db_root_path, args.num_samples)
    
    # Test instruction
    instruction = "Read the question and output only the definitional rules that apply."
    
    # Test the model
    test_model(
        model, 
        tokenizer, 
        test_samples, 
        args.db_root_path,
        instruction,
        args.max_new_tokens,
        args.temperature,
        args.do_sample
    )
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Model: {args.base_model}")
    print(f"🔧 Adapter: {args.adapter_path}")


if __name__ == "__main__":
    main()
