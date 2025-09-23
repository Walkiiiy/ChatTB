#!/usr/bin/env python
"""
Batch testing script for the fine-tuned Qwen3 model on Bird_dev dataset.

This script performs comprehensive evaluation on the Bird_dev rules_res_type.json dataset
with detailed metrics and analysis.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from Process_model.SchemaInformation import SchemaInformation


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for batch testing."""
    parser = argparse.ArgumentParser(description="Batch test fine-tuned Qwen3 model on Bird_dev")
    
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model directory")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to the fine-tuned adapter directory")
    parser.add_argument("--test_rules_file", type=str, 
                       default="./Bird_dev/rules_res_type.json",
                       help="Path to Bird_dev rules JSON file for testing")
    parser.add_argument("--db_root_path", type=str, default="./Bird_dev/dev_databases",
                       help="Root directory containing Bird_dev database files")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to test (0 for all)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for testing")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during generation")
    parser.add_argument("--output_file", type=str, default="./test_results.json",
                       help="Output file for detailed results")
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


def load_test_samples(rules_file: str, max_samples: int = 0) -> List[Dict]:
    """Load test samples from the Bird_dev rules file."""
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
    
    # Limit samples if requested
    if max_samples > 0 and len(valid_items) > max_samples:
        valid_items = valid_items[:max_samples]
    
    print(f"✅ Loaded {len(valid_items)} test samples with definitional rules")
    return valid_items


def extract_definitional_rules(rules: List[Dict]) -> List[Dict]:
    """Extract definitional rules from a rules list."""
    definitional_rules = []
    for rule in rules:
        if str(rule.get("type", "")).lower() in ["definitional", "definational"]:
            condition = rule.get("condition", "").strip()
            operation = rule.get("operation", "").strip()
            if condition and operation:
                definitional_rules.append({
                    "condition": condition,
                    "operation": operation
                })
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
            num_beams=1,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def parse_generated_rules(response: str) -> List[Dict]:
    """Parse generated rules from model response."""
    rules = []
    
    # Try to extract rules using regex patterns
    # Pattern for "Rule X:" format
    rule_pattern = r'Rule\s+\d+:\s*(.*?)(?=Rule\s+\d+:|$)'
    rule_matches = re.findall(rule_pattern, response, re.DOTALL | re.IGNORECASE)
    
    for rule_text in rule_matches:
        rule_text = rule_text.strip()
        if not rule_text:
            continue
            
        # Extract condition and operation
        condition_match = re.search(r'Condition:\s*(.*?)(?=Operation:|$)', rule_text, re.DOTALL | re.IGNORECASE)
        operation_match = re.search(r'Operation:\s*(.*?)(?=Condition:|$)', rule_text, re.DOTALL | re.IGNORECASE)
        
        if condition_match and operation_match:
            condition = condition_match.group(1).strip()
            operation = operation_match.group(1).strip()
            if condition and operation:
                rules.append({
                    "condition": condition,
                    "operation": operation
                })
    
    # If no structured rules found, try simpler patterns
    if not rules:
        # Look for any condition/operation pairs
        condition_pattern = r'condition:\s*(.*?)(?=operation:|$)'
        operation_pattern = r'operation:\s*(.*?)(?=condition:|$)'
        
        conditions = re.findall(condition_pattern, response, re.DOTALL | re.IGNORECASE)
        operations = re.findall(operation_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for cond, op in zip(conditions, operations):
            if cond.strip() and op.strip():
                rules.append({
                    "condition": cond.strip(),
                    "operation": op.strip()
                })
    
    return rules


def calculate_rule_metrics(generated_rules: List[Dict], ground_truth_rules: List[Dict]) -> Dict:
    """Calculate evaluation metrics for rule extraction."""
    metrics = {
        "num_generated": len(generated_rules),
        "num_ground_truth": len(ground_truth_rules),
        "exact_match": 0,
        "partial_match": 0,
        "condition_match": 0,
        "operation_match": 0
    }
    
    # Check for exact matches
    for gt_rule in ground_truth_rules:
        for gen_rule in generated_rules:
            if (gt_rule["condition"].lower().strip() == gen_rule["condition"].lower().strip() and
                gt_rule["operation"].lower().strip() == gen_rule["operation"].lower().strip()):
                metrics["exact_match"] += 1
                break
    
    # Check for partial matches (either condition or operation matches)
    for gt_rule in ground_truth_rules:
        for gen_rule in generated_rules:
            cond_match = gt_rule["condition"].lower().strip() == gen_rule["condition"].lower().strip()
            op_match = gt_rule["operation"].lower().strip() == gen_rule["operation"].lower().strip()
            
            if cond_match:
                metrics["condition_match"] += 1
            if op_match:
                metrics["operation_match"] += 1
            if cond_match or op_match:
                metrics["partial_match"] += 1
                break
    
    return metrics


def batch_test_model(model, tokenizer, test_samples: List[Dict], db_root_path: str, 
                    instruction: str, max_new_tokens: int, temperature: float, 
                    do_sample: bool, output_file: str):
    """Perform batch testing on all samples."""
    schema_helper = SchemaInformation()
    
    print(f"\n🧪 Starting batch testing on {len(test_samples)} samples...")
    print("=" * 80)
    
    results = []
    total_metrics = {
        "num_generated": 0,
        "num_ground_truth": 0,
        "exact_match": 0,
        "partial_match": 0,
        "condition_match": 0,
        "operation_match": 0,
        "samples_with_rules": 0,
        "samples_without_rules": 0,
        "total_time": 0
    }
    
    start_time = time.time()
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Processing sample {i}/{len(test_samples)}")
        
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
        
        # Generate model response
        sample_start = time.time()
        response = generate_response(model, tokenizer, prompt, max_new_tokens, 
                                   temperature, do_sample)
        sample_time = time.time() - sample_start
        
        # Parse generated rules
        generated_rules = parse_generated_rules(response)
        
        # Calculate metrics
        metrics = calculate_rule_metrics(generated_rules, ground_truth_rules)
        
        # Update total metrics
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]
        
        if generated_rules:
            total_metrics["samples_with_rules"] += 1
        else:
            total_metrics["samples_without_rules"] += 1
        
        total_metrics["total_time"] += sample_time
        
        # Store detailed result
        result = {
            "sample_id": i,
            "question": question,
            "db_id": db_id,
            "ground_truth_rules": ground_truth_rules,
            "generated_rules": generated_rules,
            "model_response": response,
            "metrics": metrics,
            "processing_time": sample_time
        }
        results.append(result)
        
        # Print progress
        if i % 10 == 0 or i == len(test_samples):
            print(f"Progress: {i}/{len(test_samples)} | "
                  f"Exact match: {total_metrics['exact_match']}/{total_metrics['num_ground_truth']} | "
                  f"Avg time: {total_metrics['total_time']/i:.2f}s")
    
    total_time = time.time() - start_time
    
    # Calculate final metrics
    final_metrics = {
        "total_samples": len(test_samples),
        "total_processing_time": total_time,
        "avg_processing_time": total_time / len(test_samples),
        "exact_match_rate": total_metrics["exact_match"] / max(total_metrics["num_ground_truth"], 1),
        "partial_match_rate": total_metrics["partial_match"] / max(total_metrics["num_ground_truth"], 1),
        "condition_match_rate": total_metrics["condition_match"] / max(total_metrics["num_ground_truth"], 1),
        "operation_match_rate": total_metrics["operation_match"] / max(total_metrics["num_ground_truth"], 1),
        "samples_with_rules_rate": total_metrics["samples_with_rules"] / len(test_samples),
        "avg_rules_per_sample": total_metrics["num_generated"] / len(test_samples),
        "avg_ground_truth_rules": total_metrics["num_ground_truth"] / len(test_samples)
    }
    
    # Save results
    output_data = {
        "final_metrics": final_metrics,
        "total_metrics": total_metrics,
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n🎉 Batch testing completed!")
    print("=" * 80)
    print(f"📊 Final Results:")
    print(f"  Total samples: {final_metrics['total_samples']}")
    print(f"  Total time: {final_metrics['total_processing_time']:.2f}s")
    print(f"  Avg time per sample: {final_metrics['avg_processing_time']:.2f}s")
    print(f"  Exact match rate: {final_metrics['exact_match_rate']:.3f}")
    print(f"  Partial match rate: {final_metrics['partial_match_rate']:.3f}")
    print(f"  Condition match rate: {final_metrics['condition_match_rate']:.3f}")
    print(f"  Operation match rate: {final_metrics['operation_match_rate']:.3f}")
    print(f"  Samples with rules: {final_metrics['samples_with_rules_rate']:.3f}")
    print(f"  Avg generated rules: {final_metrics['avg_rules_per_sample']:.2f}")
    print(f"  Avg ground truth rules: {final_metrics['avg_ground_truth_rules']:.2f}")
    print(f"\n📁 Detailed results saved to: {output_file}")


def main():
    """Main function to run the batch test."""
    print("🚀 Starting Batch Model Test on Bird_dev Dataset")
    print("=" * 60)
    
    args = parse_args()
    
    # Validate paths
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"Base model path not found: {args.base_model}")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if not os.path.exists(args.test_rules_file):
        raise FileNotFoundError(f"Test rules file not found: {args.test_rules_file}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, 
        args.adapter_path, 
        args.trust_remote_code
    )
    
    # Load test samples
    test_samples = load_test_samples(args.test_rules_file, args.max_samples)
    
    if not test_samples:
        print("❌ No valid test samples found!")
        return
    
    # Test instruction
    instruction = "Read the question and output only the definitional rules that apply."
    
    # Run batch test
    batch_test_model(
        model, 
        tokenizer, 
        test_samples, 
        args.db_root_path,
        instruction,
        args.max_new_tokens,
        args.temperature,
        args.do_sample,
        args.output_file
    )


if __name__ == "__main__":
    main()



