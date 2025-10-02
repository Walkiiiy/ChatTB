#!/usr/bin/env python
"""
Rule Fixing Script - Load and Fix Pre-generated Rules

This script implements a rule fixing approach:
1. Load pre-generated rules from generated_rules.json
2. Process condensed_rules.json dataset
3. For each generated rule, calculate embedding similarity with dataset rules
4. Find the most similar dataset rule (similarity > 0.7)
5. Use model B to modify the generated rule to achieve the same effect as the dataset rule
6. Replace the original rule with the modified rule
7. Output fixed rules to fixed_rules.json

Key Features:
- Load pre-generated rules instead of generating them
- Semantic similarity matching using Process_model/SemanticSimilarity.py
- Rule modification using Process_model/LLMClient.py or DeepSeekLLMClient.py
- Rule fixing and enhancement pipeline

Usage:
    python AssumerInferenceFixing.py --model_b /path/to/modification/model --dataset /path/to/condensed_rules.json --generated_rules /path/to/generated_rules.json --output_dir /path/to/output
"""

import argparse
import json
import random
import os
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Removed model A related imports as they are no longer needed

# Import our custom classes
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation
from Process_model.SemanticSimilarity import SemanticSimilarity
from Process_model.LLMClient import LLMClient
from Process_model.DeepSeekLLMClient import DeepSeekLLMClient

from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for rule fixing."""
    parser = argparse.ArgumentParser(description="Rule Fixing - Load and Fix Pre-generated Rules")
    
    # Model configuration
    parser.add_argument("--model_b", type=str, required=True,
                       help="Path to model B (for rule modification)")
    parser.add_argument("--model_b_type", type=str, choices=["local", "deepseek"], default="local",
                       help="Type of model B: 'local' for local model, 'deepseek' for DeepSeek API")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of custom modeling code")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to condensed_rules.json dataset file")
    parser.add_argument("--generated_rules", type=str, required=True,
                       help="Path to generated_rules.json file containing pre-generated rules")
    parser.add_argument("--db_root_path", type=str, required=True,
                       help="Root directory containing database files")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to process. Use -1 to process all samples.")
    parser.add_argument("--output_dir", type=str, default="dual_sft_results",
                       help="Output directory for results and fine-tuned model")
    
    # Similarity and matching parameters
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="Minimum similarity threshold for rule matching")
    parser.add_argument("--similarity_method", type=str, default="embedding",
                       choices=["embedding", "tfidf", "jaccard", "levenshtein"],
                       help="Method for calculating similarity")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate for rules")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for rule generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during rule generation")
    
    # Output configuration
    parser.add_argument("--output_dataset", type=str, default="fixed_rules.json",
                       help="Output file for the fixed rules dataset")
    
    # GPU configuration
    parser.add_argument("--model_b_gpu", type=int, default=0,
                       help="GPU ID for model B (default: 0)")
    
    # DeepSeek API configuration (if using model_b_type=deepseek)
    parser.add_argument("--deepseek_api_key", type=str, default=None,
                       help="DeepSeek API key (required if model_b_type=deepseek)")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-coder",
                       help="DeepSeek model name")
    
    return parser.parse_args()




def load_model_b(model_path: str, model_type: str, trust_remote_code: bool, api_key: str = None, gpu_id: int = 0):
    """Load model B for rule modification."""
    logger = logging.getLogger(__name__)
    
    if model_type == "local":
        logger.info(f"📖 Loading local model B from: {model_path} on GPU {gpu_id}")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {gpu_id} is not available! Available GPUs: {torch.cuda.device_count()}")
        
        logger.info(f"🎯 Using device: cuda:{gpu_id}")
        
        # Create LLMClient with specific GPU
        client = LLMClient(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            max_new_tokens=256,
            temperature=0.1
        )
        
        # Move model to specific GPU if it's a local model
        if hasattr(client, 'model') and client.model is not None:
            client.model = client.model.to(f"cuda:{gpu_id}")
            logger.info(f"✅ Model B loaded successfully on GPU {gpu_id}!")
            logger.info(f"📊 GPU {gpu_id} memory usage: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
        
        return client
        
    elif model_type == "deepseek":
        if not api_key:
            raise ValueError("DeepSeek API key is required when using model_b_type=deepseek")
        logger.info(f"🌐 Initializing DeepSeek model B")
        return DeepSeekLLMClient(
            api_key=api_key,
            model="deepseek-coder",
            max_tokens=256,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported model_b_type: {model_type}")


def split_rules_into_individual(rules_text: str) -> List[str]:
    """Split rules text into individual rules."""
    if not rules_text or not rules_text.strip():
        return []
    
    # Split by common rule separators
    rules = []
    
    # Try splitting by numbered lists (1., 2., etc.)
    numbered_rules = re.split(r'\n\s*\d+\.\s*', rules_text)
    if len(numbered_rules) > 1:
        for rule in numbered_rules[1:]:  # Skip the first empty part
            rule = rule.strip()
            if rule:
                rules.append(rule)
        return rules
    
    # Try splitting by line breaks and filter out empty lines
    lines = rules_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('rules:'):
            # Remove common prefixes
            line = re.sub(r'^\d+\)\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            line = re.sub(r'^\*\s*', '', line)
            if line:
                rules.append(line)
    
    return rules


def build_rule_modification_prompt(original_rule: str, target_rule: str, question: str, schema: str) -> str:
    """Build prompt for rule modification using model B."""
    prompt = f"""You are an expert in database rule refinement. Your task is to modify a rule to achieve the same effect as a target rule while making minimal changes.

Original Rule: {original_rule}

Target Rule (for reference): {target_rule}


Database Schema: {schema}

Task: Modify the original rule to achieve the same effect as the target rule. Make minimal changes while ensuring the rule is:
1. Schema-faithful (uses correct table/column names)
2. Concise and clear
3. Achieves the same logical effect as the target rule

Modified Rule:"""
    
    return prompt




class RuleFixingProcessor:
    """Main class for rule fixing pipeline."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the rule fixing processor."""
        self.args = args
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize model B as None (will be loaded when needed)
        self.model_b = None
        
        # Initialize components
        self.similarity_calculator = SemanticSimilarity(use_gpu=False)
        self.schema_helper = SchemaInformation()
        
        # Load dataset
        self.dataset = self._load_dataset(args.dataset, args.num_samples)
        
        # Load generated rules
        self.generated_rules = self._load_generated_rules(args.generated_rules)
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "rules_loaded": 0,
            "rules_modified": 0,
            "similarity_matches": 0,
            "no_similarity_matches": 0
        }
        
        self.logger.info("Rule Fixing Processor initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        log_file = self.output_dir / "rule_fixing.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear root logger handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
    
    def _load_dataset(self, dataset_path: str, num_samples: int) -> List[Dict]:
        """Load dataset samples."""
        self.logger.info(f"📋 Loading dataset from: {dataset_path}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to list if it's a dict
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data
        
        # Filter items that have rules
        items_with_rules = []
        for item in items:
            rules = item.get("rules", [])
            if rules and len(rules) > 0:
                items_with_rules.append(item)
        
        self.logger.info(f"📊 Found {len(items_with_rules)} samples with rules out of {len(items)} total samples")
        
        if len(items_with_rules) == 0:
            self.logger.warning("⚠️  No samples with rules found in the dataset!")
            return []
        
        # Handle -1 as "process all samples"
        if num_samples == -1:
            sampled_items = items_with_rules
            self.logger.info(f"✅ Processing ALL {len(sampled_items)} samples with rules")
        else:
            sampled_items = random.sample(items_with_rules, min(num_samples, len(items_with_rules)))
            self.logger.info(f"✅ Selected {len(sampled_items)} samples for processing")
        
        return sampled_items
    
    def _load_generated_rules(self, generated_rules_path: str) -> Dict:
        """Load generated rules from JSON file."""
        self.logger.info(f"📋 Loading generated rules from: {generated_rules_path}")
        
        with open(generated_rules_path, "r", encoding="utf-8") as f:
            generated_rules = json.load(f)
        
        self.logger.info(f"📊 Loaded {len(generated_rules)} generated rules")
        return generated_rules
    
    
    def load_model_b(self):
        """Load model B for rule modification."""
        if self.model_b is None:
            self.logger.info("🔄 Loading model B...")
            self.model_b = load_model_b(
                self.args.model_b, self.args.model_b_type, 
                self.args.trust_remote_code, self.args.deepseek_api_key, self.args.model_b_gpu
            )
            self.logger.info("✅ Model B loaded successfully")
    
    def unload_model_b(self):
        """Unload model B to free memory."""
        if self.model_b is not None:
            self.logger.info("🔄 Unloading model B...")
            del self.model_b
            self.model_b = None
            torch.cuda.empty_cache()  # Clear GPU cache
            self.logger.info("✅ Model B unloaded successfully")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        self.logger.info("🔄 Ensuring all models are unloaded...")
        self.unload_model_b()
        self.logger.info("✅ All models unloaded successfully")
    
    def print_gpu_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            self.logger.info("📊 GPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        else:
            self.logger.warning("CUDA is not available!")
    
    
    def find_best_matching_rule(self, generated_rule: str, dataset_rules: List[str]) -> Tuple[Optional[str], float]:
        """Find the best matching dataset rule for a generated rule."""
        best_rule = None
        best_similarity = 0.0
        
        for dataset_rule in dataset_rules:
            similarity_result = self.similarity_calculator.calculate_similarity(
                generated_rule, dataset_rule, self.args.similarity_method
            )
            
            if isinstance(similarity_result, dict):
                similarity = similarity_result.get("similarity", 0.0)
            else:
                similarity = similarity_result
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_rule = dataset_rule
        
        return best_rule, best_similarity
    
    def modify_rule_with_model_b(self, original_rule: str, target_rule: str, question: str, schema: str) -> str:
        """Modify a rule using model B."""
        prompt = build_rule_modification_prompt(original_rule, target_rule, question, schema)
        
        if self.args.model_b_type == "local":
            response = self.model_b.chat(prompt)
        else:  # deepseek
            response = self.model_b.chat(prompt)
        
        return response.strip()
    
    def process_single_sample(self, sample: Dict, sample_idx: int) -> Dict:
        """Process a single sample: load rules from generated_rules.json, then modify with model B."""
        question = sample.get("question", "").strip()
        db_id = sample.get("db_id", "").strip()
        dataset_rules = sample.get("rules", [])
        
        self.logger.info(f"📝 Processing sample {sample_idx + 1} - {db_id}")
        self.logger.info(f"  Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        result = {
            "sample_idx": sample_idx,
            "question": question,
            "db_id": db_id,
            "original_generated_rules": "",
            "modified_rules": "",
            "dataset_rules": dataset_rules,
            "schema": "",
            "rule_modifications": [],
            "error_message": ""
        }
        
        try:
            # Get database schema
            schema = ""
            if db_id:
                db_path = os.path.join(self.args.db_root_path, db_id, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    try:
                        self.logger.info(f"  🔍 Generating schema for {db_id}...")
                        schema = self.schema_helper.generate_schema_info(db_path)
                        self.logger.info(f"  ✅ Schema generated ({len(schema)} characters)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to generate schema for {db_id}: {e}")
                        result["error_message"] = f"Schema generation failed: {e}"
                        return result
                else:
                    self.logger.error(f"❌ Database file not found: {db_path}")
                    result["error_message"] = f"Database file not found: {db_path}"
                    return result
            
            if not schema:
                self.logger.error("❌ No schema found for this database")
                result["error_message"] = "No schema found"
                return result
            
            # Load rules from generated_rules.json
            self.logger.info(f"  📖 Loading rules from generated_rules.json...")
            sample_key = str(sample_idx)
            if sample_key in self.generated_rules:
                generated_rules_data = self.generated_rules[sample_key]
                generated_rules = generated_rules_data.get("rules", "")
                if isinstance(generated_rules, list):
                    generated_rules = "\n".join(generated_rules)
                result["original_generated_rules"] = generated_rules
                result["schema"] = schema
                self.stats["rules_loaded"] += 1
                self.logger.info(f"  ✅ Loaded rules from generated_rules.json")
            else:
                self.logger.error(f"  ❌ No generated rules found for sample {sample_idx}")
                result["error_message"] = f"No generated rules found for sample {sample_idx}"
                return result
            
            individual_rules = split_rules_into_individual(generated_rules)
            self.logger.info(f"  ✅ Loaded {len(individual_rules)} rules")
            
            # Display generated rules (truncated for readability)
            for j, rule in enumerate(individual_rules[:3]):  # Show first 3 rules
                self.logger.info(f"    Rule {j+1}: {rule[:80]}{'...' if len(rule) > 80 else ''}")
            if len(individual_rules) > 3:
                self.logger.info(f"    ... and {len(individual_rules) - 3} more rules")
            
            # Modify rules using model B
            self.logger.info(f"  🔧 Modifying rules with Model B...")
            modified_rules = []
            rule_modifications = []
            
            for j, generated_rule in enumerate(individual_rules):
                self.logger.info(f"    [{j+1}/{len(individual_rules)}] Processing rule: {generated_rule[:60]}{'...' if len(generated_rule) > 60 else ''}")
                
                # Find best matching dataset rule
                best_match, similarity = self.find_best_matching_rule(generated_rule, dataset_rules)
                
                self.logger.info(f"      🎯 Best match similarity: {similarity:.3f} (threshold: {self.args.similarity_threshold})")
                
                if similarity >= self.args.similarity_threshold:
                    self.stats["similarity_matches"] += 1
                    self.logger.info(f"      ✅ Similarity above threshold - modifying rule...")
                    
                    # Modify the rule using model B
                    try:
                        modified_rule = self.modify_rule_with_model_b(
                            generated_rule, best_match, question, schema
                        )
                        
                        rule_modifications.append({
                            "original": generated_rule,
                            "target": best_match,
                            "modified": modified_rule,
                            "similarity": similarity
                        })
                        
                        modified_rules.append(modified_rule)
                        self.stats["rules_modified"] += 1
                        
                        self.logger.info(f"      🔄 Rule modified successfully")
                        self.logger.info(f"        Original: {generated_rule[:50]}{'...' if len(generated_rule) > 50 else ''}")
                        self.logger.info(f"        Modified: {modified_rule[:50]}{'...' if len(modified_rule) > 50 else ''}")
                        
                    except Exception as e:
                        self.logger.error(f"      ❌ Failed to modify rule: {e}")
                        modified_rules.append(generated_rule)  # Keep original if modification fails
                else:
                    self.stats["no_similarity_matches"] += 1
                    modified_rules.append(generated_rule)  # Keep original rule
                    self.logger.info(f"      ⏭️  Similarity below threshold - keeping original rule")
            
            result["modified_rules"] = "\n".join(modified_rules)
            result["rule_modifications"] = rule_modifications
            
            # Summary for this sample
            modifications_count = len(rule_modifications)
            self.logger.info(f"  📈 Sample summary: {modifications_count}/{len(individual_rules)} rules modified")
            
            self.stats["total_samples"] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error processing sample: {e}")
            result["error_message"] = str(e)
        
        return result
    
    
    
    def run_rule_fixing_pipeline(self):
        """Run the rule fixing pipeline: Load rules from generated_rules.json -> ModelB modifies rules -> Output to JSON."""
        self.logger.info(f"\n🚀 Starting Rule Fixing Pipeline")
        self.logger.info(f"Processing {len(self.dataset)} samples...")
        self.logger.info("=" * 80)
        
        # Ensure all models are unloaded at the start
        self.unload_all_models()
        
        # Print initial GPU status
        self.print_gpu_memory_usage()
        
        # Load model B only (for rule modification)
        self.logger.info("🤖 Loading model B for rule modification...")
        self.load_model_b()
        
        all_results = []
        
        # Process all samples
        for i, sample in enumerate(self.dataset):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📝 Processing sample {i+1}/{len(self.dataset)}")
            self.logger.info(f"{'='*60}")
            
            result = self.process_single_sample(sample, i)
            all_results.append(result)
            
            # Progress update
            progress = ((i + 1) / len(self.dataset)) * 100
            self.logger.info(f"📈 Overall progress: {progress:.1f}% ({i+1}/{len(self.dataset)} samples)")
        
        # Unload model B
        self.unload_model_b()
        
        # Save results
        self.save_fixed_rules_dataset(all_results)
        
        # Print final statistics
        self.print_final_statistics()
    
    def save_fixed_rules_dataset(self, results: List[Dict]):
        """Save the fixed rules dataset in the same format as the input dataset."""
        try:
            # Create the modified dataset
            modified_dataset = []
            
            for result in results:
                if result.get("error_message"):
                    self.logger.warning(f"Skipping sample {result['sample_idx'] + 1} due to error: {result['error_message']}")
                    continue
                
                # Create new sample with modified rules
                modified_sample = {
                    "db_id": result["db_id"],
                    "question": result["question"],
                    "ground_truth": result.get("ground_truth", ""),  # Keep original if exists
                    "amends": result.get("amends", ""),  # Keep original if exists
                    "rules": split_rules_into_individual(result["modified_rules"])  # Convert to list format
                }
                
                modified_dataset.append(modified_sample)
            
            # Save the modified dataset
            output_file = self.output_dir / self.args.output_dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(modified_dataset, f, indent=2, ensure_ascii=False)
            
            # Also save detailed results for analysis
            results_file = self.output_dir / "detailed_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_samples": len(results),
                        "args": vars(self.args)
                    },
                    "statistics": self.stats,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Fixed rules dataset saved to: {output_file}")
            self.logger.info(f"✅ Detailed results saved to: {results_file}")
            self.logger.info(f"📊 Dataset contains {len(modified_dataset)} samples with fixed rules")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save fixed rules dataset: {e}")
            raise
    
    
    
    def print_final_statistics(self):
        """Print final processing statistics."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("RULE FIXING PIPELINE COMPLETED")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total Samples Processed: {self.stats['total_samples']}")
        self.logger.info(f"Rules Loaded: {self.stats['rules_loaded']}")
        self.logger.info(f"Rules Modified: {self.stats['rules_modified']}")
        self.logger.info(f"Similarity Matches: {self.stats['similarity_matches']}")
        self.logger.info(f"No Similarity Matches: {self.stats['no_similarity_matches']}")
        
        if self.stats['rules_loaded'] > 0:
            modification_rate = self.stats['rules_modified'] / self.stats['rules_loaded']
            match_rate = self.stats['similarity_matches'] / self.stats['rules_loaded']
            
            self.logger.info(f"\n📊 STATISTICS:")
            self.logger.info(f"  Rule Modification Rate: {modification_rate:.3f} ({modification_rate*100:.1f}%)")
            self.logger.info(f"  Similarity Match Rate: {match_rate:.3f} ({match_rate*100:.1f}%)")
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")


def main():
    """Main function to run the dual SFT pipeline."""
    logger = logging.getLogger(__name__)
    
    # Set up basic logging first (will be overridden by DualSFTProcessor)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Rule Fixing Pipeline")
    logger.info("=" * 60)
    
    args = parse_args()
    
    # Validate paths and arguments
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if not os.path.exists(args.generated_rules):
        raise FileNotFoundError(f"Generated rules file not found: {args.generated_rules}")
    if not os.path.isdir(args.db_root_path):
        raise FileNotFoundError(f"Database root path not found: {args.db_root_path}")
    
    # Validate model B configuration
    if args.model_b_type == "local":
        if not os.path.isdir(args.model_b):
            raise FileNotFoundError(f"Model B path not found: {args.model_b}")
    elif args.model_b_type == "deepseek":
        if not args.deepseek_api_key:
            raise ValueError("DeepSeek API key is required when using model_b_type=deepseek")
    
    try:
        # Initialize rule fixing processor
        processor = RuleFixingProcessor(args)
        
        # Run the rule fixing pipeline
        processor.run_rule_fixing_pipeline()
        
        logger.info("✅ Rule fixing pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()