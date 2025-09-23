#!/usr/bin/env python
"""
Quick test script for Bird_dev dataset - tests only a few samples for rapid evaluation.
"""

import json
import random
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from Process_model.SchemaInformation import SchemaInformation


def load_model_and_tokenizer():
    """Load the fine-tuned model quickly."""
    print("📖 Loading fine-tuned model...")
    
    base_model_path = "/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
    adapter_path = "/home/ubuntu/walkiiiy/ChatTB/Process_model/models--globalAssumer_Qwen3_8b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Model loaded successfully!")
    
    return model, tokenizer


def test_sample(model, tokenizer, sample: Dict, db_root_path: str) -> Dict:
    """Test a single sample."""
    schema_helper = SchemaInformation()
    
    # Extract sample info
    question = sample.get("question", "").strip()
    db_id = sample.get("db_id", "").strip()
    rules = sample.get("rules", [])
    
    # Get ground truth definitional rules
    ground_truth_rules = []
    for rule in rules:
        if str(rule.get("type", "")).lower() in ["definitional", "definational"]:
            condition = rule.get("condition", "").strip()
            operation = rule.get("operation", "").strip()
            if condition and operation:
                ground_truth_rules.append({
                    "condition": condition,
                    "operation": operation
                })
    
    # Generate schema
    schema = ""
    if db_id:
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_path):
            try:
                schema = schema_helper.generate_schema_info(db_path)
            except Exception:
                schema = ""
    
    # Build prompt
    instruction = "Read the question and output only the definitional rules that apply."
    prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"definitional rules:\n"
    )
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return {
        "question": question,
        "db_id": db_id,
        "ground_truth_rules": ground_truth_rules,
        "model_response": response,
        "num_ground_truth": len(ground_truth_rules),
        "response_length": len(response)
    }


def main():
    """Quick test on Bird_dev dataset."""
    print("🚀 Quick Bird_dev Test")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Load test data
    rules_file = "/home/ubuntu/walkiiiy/ChatTB/Bird_dev/rules_res_type.json"
    db_root_path = "/home/ubuntu/walkiiiy/ChatTB/Bird_dev/dev_databases"
    
    with open(rules_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get valid samples
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    
    # Filter samples with definitional rules
    valid_items = []
    for item in items:
        rules = item.get("rules", [])
        has_definitional = any(
            str(rule.get("type", "")).lower() in ["definitional", "definational"]
            for rule in rules
        )
        if has_definitional:
            valid_items.append(item)
    
    # Test 5 random samples
    test_samples = random.sample(valid_items, min(5, len(valid_items)))
    
    print(f"Testing {len(test_samples)} samples...")
    
    results = []
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Sample {i}/{len(test_samples)}")
        print("-" * 40)
        
        result = test_sample(model, tokenizer, sample, db_root_path)
        results.append(result)
        
        print(f"Question: {result['question']}")
        print(f"Database: {result['db_id']}")
        print(f"Ground truth rules: {result['num_ground_truth']}")
        print(f"Model response length: {result['response_length']} chars")
        
        print(f"\nGround Truth Rules:")
        for j, rule in enumerate(result['ground_truth_rules'], 1):
            print(f"  {j}. Condition: {rule['condition']}")
            print(f"     Operation: {rule['operation']}")
        
        print(f"\nModel Response:")
        print(f"{result['model_response']}")
        print("-" * 40)
    
    # Summary
    total_gt_rules = sum(r['num_ground_truth'] for r in results)
    avg_response_length = sum(r['response_length'] for r in results) / len(results)
    
    print(f"\n📊 Quick Test Summary:")
    print(f"  Samples tested: {len(results)}")
    print(f"  Total ground truth rules: {total_gt_rules}")
    print(f"  Average response length: {avg_response_length:.1f} chars")
    print(f"  Avg rules per sample: {total_gt_rules/len(results):.1f}")


if __name__ == "__main__":
    main()



