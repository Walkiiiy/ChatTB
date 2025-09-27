#!/usr/bin/env python
"""
SFT Results Test Script - Local Model SQL Generation Version

This script tests the SFT fine-tuned model by comparing THREE approaches:
1. Loading the fine-tuned model with LoRA adapter for rule generation
2. Loading a separate local model for SQL generation
3. Randomly selecting samples from a dataset (supports condensed_rules.json format)
   NOTE: Only samples containing rules will be selected for testing
4. Using the SFT model to generate rules for each question
5. Testing THREE approaches:
   A. Generate SQL using local model WITH SFT-generated rules
   B. Generate SQL using local model WITH dataset rules (ground truth)
   C. Generate SQL using local model WITHOUT rules (baseline)
6. Validating all three generated SQLs against ground truth
7. Comparing performance to measure the impact of different rule sources

This allows you to evaluate:
- Whether SFT model's generated rules improve SQL generation performance
- How SFT-generated rules compare to ground truth rules
- The overall effectiveness of rules-based SQL generation

Enhanced with comprehensive reporting:
- CSV reports with detailed results and summary statistics
- Visualization plots showing accuracy comparisons and improvements
- Enhanced JSON reports with detailed analysis
- Human-readable summary reports

Usage:
    python AssumerSFTTest_LocalModelSQL.py --base_model /path/to/base/model --adapter_path /path/to/adapter --sql_model /path/to/sql/model --dataset /path/to/condensed_rules.json --num_samples 10
"""

import argparse
import json
import random
import os
import logging
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import our custom classes
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation

from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SFT results testing."""
    parser = argparse.ArgumentParser(description="Test SFT fine-tuned model results")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model directory for rule generation")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to the fine-tuned adapter directory")
    parser.add_argument("--sql_model", type=str, required=True,
                       help="Path to the local model directory for SQL generation")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of custom modeling code")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True,
                       help="Root directory containing database files")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of random samples to test (only samples with rules will be selected)")
    parser.add_argument("--output_dir", type=str, default="sft_test_results",
                       help="Output directory for test results and reports")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate for rules")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for rule generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during rule generation")
    
    # SQL generation parameters
    parser.add_argument("--sql_max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate for SQL")
    parser.add_argument("--sql_temperature", type=float, default=0.1,
                       help="Temperature for SQL generation")
    parser.add_argument("--sql_do_sample", action="store_true",
                       help="Enable sampling during SQL generation")
    
    return parser.parse_args()


def load_model_and_tokenizer(base_model_path: str, adapter_path: str, trust_remote_code: bool):
    """Load the base model and fine-tuned adapter for rule generation."""
    logger = logging.getLogger(__name__)
    logger.info(f"📖 Loading base model from: {base_model_path}")
    
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
    logger.info(f"🔧 Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("✅ Rule generation model and adapter loaded successfully!")
    return model, tokenizer


def load_sql_model_and_tokenizer(sql_model_path: str, trust_remote_code: bool):
    """Load the SQL generation model and tokenizer."""
    logger = logging.getLogger(__name__)
    logger.info(f"📖 Loading SQL model from: {sql_model_path}")
    
    # Load tokenizer
    sql_tokenizer = AutoTokenizer.from_pretrained(
        sql_model_path, 
        use_fast=True, 
        trust_remote_code=trust_remote_code
    )
    if sql_tokenizer.pad_token is None:
        sql_tokenizer.pad_token = sql_tokenizer.eos_token
    sql_tokenizer.padding_side = "right"
    
    # Load SQL model
    sql_model = AutoModelForCausalLM.from_pretrained(
        sql_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    
    logger.info("✅ SQL generation model loaded successfully!")
    return sql_model, sql_tokenizer


def load_test_samples(dataset_path: str, num_samples: int) -> List[Dict]:
    """Load random samples from the dataset for testing, only including samples with rules."""
    logger = logging.getLogger(__name__)
    logger.info(f"📋 Loading test samples from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to list if it's a dict
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    
    # Filter items that have rules (new condensed_rules format)
    items_with_rules = []
    for item in items:
        rules = item.get("rules", [])
        
        # Check if there are any rules (simple string array format)
        if rules and len(rules) > 0:
            items_with_rules.append(item)
    
    logger.info(f"📊 Found {len(items_with_rules)} samples with rules out of {len(items)} total samples")
    
    if len(items_with_rules) == 0:
        logger.warning("⚠️  No samples with rules found in the dataset!")
        return []
    
    # Sample random items from those with rules
    sampled_items = random.sample(items_with_rules, min(num_samples, len(items_with_rules)))
    
    logger.info(f"✅ Selected {len(sampled_items)} test samples (all with rules)")
    return sampled_items


def extract_rules_from_dataset(rules: List[str]) -> str:
    """Extract rules from dataset rules list (new condensed_rules format)."""
    if not rules:
        return "No rules found."
    
    # Rules are now simple strings, just join them
    return "\n\n".join([f"{i+1}. {rule.strip()}" for i, rule in enumerate(rules) if rule.strip()])


def build_rule_generation_prompt(instruction: str, schema: str, question: str) -> str:
    """Build prompt for rule generation."""
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"generated rules:\n"
    )


def generate_rules(model, tokenizer, prompt: str, max_new_tokens: int, 
                  temperature: float, do_sample: bool) -> str:
    """Generate rules from the fine-tuned model."""
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


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int, 
                temperature: float, do_sample: bool) -> str:
    """Generate SQL from the local model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
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


def build_sql_generation_prompt_with_rules(question: str, schema: str, rules: str) -> str:
    """Build prompt for SQL generation with rules."""
    prompt = f"""You are a helpful assistant that writes valid SQLite queries.

You will be given database schema, a question related to the database and some rules.
You should generate a SQLite query that solves the question with the help of rules.
The rules contain all the rules you should obey while generating the target SQL, you have to obey all of them.

Database Schema:
{schema}

Question: {question}

Rules: {rules}

Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.

SQL:"""
    
    return prompt


def build_sql_generation_prompt_without_rules(question: str, schema: str) -> str:
    """Build prompt for SQL generation without rules."""
    prompt = f"""You are a helpful assistant that writes valid SQLite queries.

You will be given database schema and a question related to the database.
You should generate a SQLite query that solves the question.

Database Schema:
{schema}

Question: {question}

Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.

SQL:"""
    
    return prompt


def build_sql_generation_prompt_with_dataset_rules(question: str, schema: str, dataset_rules: str) -> str:
    """Build prompt for SQL generation with dataset rules."""
    prompt = f"""You are a helpful assistant that writes valid SQLite queries.

You will be given database schema, a question related to the database and some rules.
You should generate a SQLite query that solves the question with the help of rules.
The rules contain all the rules you should obey while generating the target SQL, you have to obey all of them.

Database Schema:
{schema}

Question: {question}

Rules: {dataset_rules}

Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.

SQL:"""
    
    return prompt


def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from model response, handling various formats."""
    # Remove markdown code blocks
    response = response.strip()
    if response.startswith('```sql'):
        response = response[6:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]
    
    # Clean up whitespace
    response = response.strip()
    
    # Find SQL-like content (starts with SELECT, INSERT, UPDATE, DELETE, etc.)
    lines = response.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
            # Stop at semicolon or empty line after SQL
            if line.endswith(';') or (not line and sql_lines):
                break
    
    if sql_lines:
        sql = ' '.join(sql_lines).strip()
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    return response


class SFTResultsAnalyzer:
    """Analyzes and generates comprehensive reports for SFT test results."""
    
    def __init__(self, output_dir: str):
        """Initialize the results analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_csv_report(self, results: List[Dict], statistics: Dict) -> str:
        """Generate CSV report from test results."""
        csv_data = []
        
        for result in results:
            csv_data.append({
                'sample_idx': result['sample_idx'],
                'db_id': result['db_id'],
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'sft_rules_result': result['result_with_sft_rules'],
                'dataset_rules_result': result['result_with_dataset_rules'],
                'no_rules_result': result['result_without_rules'],
                'sft_sql': result['sql_with_sft_rules'],
                'dataset_sql': result['sql_with_dataset_rules'],
                'no_rules_sql': result['sql_without_rules'],
                'generated_rules': result['generated_rules'],
                'dataset_rules': result['dataset_rules'],
                'error_message': result.get('error_message', '')
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Generate summary statistics CSV
        summary_data = {
            'Metric': [
                'Total Samples',
                'SFT Rules - Correct',
                'SFT Rules - Failed',
                'SFT Rules - Accuracy',
                'Dataset Rules - Correct',
                'Dataset Rules - Failed',
                'Dataset Rules - Accuracy',
                'No Rules - Correct',
                'No Rules - Failed',
                'No Rules - Accuracy',
                'SFT vs No Rules Improvement',
                'Dataset vs No Rules Improvement',
                'SFT vs Dataset Difference'
            ],
            'Value': [
                statistics['total_questions'],
                statistics['with_sft_rules']['correct_answers'],
                statistics['with_sft_rules']['failed_executions'],
                f"{statistics['with_sft_rules']['accuracy']:.3f}",
                statistics['with_dataset_rules']['correct_answers'],
                statistics['with_dataset_rules']['failed_executions'],
                f"{statistics['with_dataset_rules']['accuracy']:.3f}",
                statistics['without_rules']['correct_answers'],
                statistics['without_rules']['failed_executions'],
                f"{statistics['without_rules']['accuracy']:.3f}",
                f"{statistics['improvements']['sft_vs_no_rules']['absolute']:+.3f}",
                f"{statistics['improvements']['dataset_vs_no_rules']['absolute']:+.3f}",
                f"{statistics['improvements']['sft_vs_dataset']['absolute']:+.3f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"CSV reports saved to {csv_path} and {summary_csv_path}")
        return str(csv_path)
    
    def generate_visualizations(self, results: List[Dict], statistics: Dict) -> None:
        """Generate visualization plots for the results."""
        plt.style.use('default')
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SFT Local Model SQL Generation Test Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        methods = ['SFT Rules', 'Dataset Rules', 'No Rules (Baseline)']
        accuracies = [
            statistics['with_sft_rules']['accuracy'],
            statistics['with_dataset_rules']['accuracy'],
            statistics['without_rules']['accuracy']
        ]
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']  # Green, Blue, Red
        bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Correct vs Failed Executions
        ax2 = axes[0, 1]
        correct_counts = [
            statistics['with_sft_rules']['correct_answers'],
            statistics['with_dataset_rules']['correct_answers'],
            statistics['without_rules']['correct_answers']
        ]
        failed_counts = [
            statistics['with_sft_rules']['failed_executions'],
            statistics['with_dataset_rules']['failed_executions'],
            statistics['without_rules']['failed_executions']
        ]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax2.bar(x - width/2, correct_counts, width, label='Correct', color='#2E8B57', alpha=0.8)
        ax2.bar(x + width/2, failed_counts, width, label='Failed', color='#DC143C', alpha=0.8)
        
        ax2.set_title('Correct vs Failed Executions')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45)
        ax2.legend()
        
        # 3. Improvement Analysis
        ax3 = axes[1, 0]
        improvements = [
            statistics['improvements']['sft_vs_no_rules']['absolute'],
            statistics['improvements']['dataset_vs_no_rules']['absolute'],
            statistics['improvements']['sft_vs_dataset']['absolute']
        ]
        improvement_labels = ['SFT vs No Rules', 'Dataset vs No Rules', 'SFT vs Dataset']
        
        colors_imp = ['green' if imp >= 0 else 'red' for imp in improvements]
        bars = ax3.bar(improvement_labels, improvements, color=colors_imp, alpha=0.7)
        ax3.set_title('Improvement Analysis')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if value >= 0 else -0.02),
                    f'{value:+.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')
        
        # 4. Result Distribution
        ax4 = axes[1, 1]
        result_distribution = [0, 0, 0]  # Correct, Incorrect, Failed
        
        for result in results:
            for method in ['result_with_sft_rules', 'result_with_dataset_rules', 'result_without_rules']:
                res = result[method]
                if res == 1:
                    result_distribution[0] += 1
                elif res == 0:
                    result_distribution[1] += 1
                else:
                    result_distribution[2] += 1
        
        # Average across all methods
        result_distribution = [x / (len(results) * 3) for x in result_distribution]
        labels = ['Correct', 'Incorrect', 'Failed']
        colors_dist = ['#2E8B57', '#FFD700', '#DC143C']
        
        wedges, texts, autotexts = ax4.pie(result_distribution, labels=labels, colors=colors_dist, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Result Distribution (Average)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "sft_test_results_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {plot_path}")
        
        # Generate detailed comparison plot
        self._generate_detailed_comparison_plot(results)
    
    def _generate_detailed_comparison_plot(self, results: List[Dict]) -> None:
        """Generate detailed comparison plot showing individual sample results."""
        # Create a comparison matrix
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Sample': result['sample_idx'],
                'SFT Rules': result['result_with_sft_rules'],
                'Dataset Rules': result['result_with_dataset_rules'],
                'No Rules': result['result_without_rules']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(results) * 0.3)))
        
        # Prepare data for heatmap (convert to numeric, handle -1 as 0.5 for visualization)
        heatmap_data = df.set_index('Sample').replace({1: 1, 0: 0, -1: 0.5})
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        colors = ['red', 'yellow', 'green']  # Failed, Incorrect, Correct
        n_bins = 3
        cmap = ListedColormap(colors)
        
        sns.heatmap(heatmap_data, annot=True, cmap=cmap, cbar_kws={'label': 'Result'},
                   vmin=0, vmax=1, fmt='.1f', linewidths=0.5)
        
        plt.title('Individual Sample Results Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Method')
        plt.ylabel('Sample Index')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Failed (-1)'),
            Patch(facecolor='yellow', label='Incorrect (0)'),
            Patch(facecolor='green', label='Correct (1)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_plot_path = self.output_dir / "detailed_sample_comparison.png"
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Detailed comparison plot saved to {detailed_plot_path}")
    
    def generate_enhanced_json_report(self, results: List[Dict], statistics: Dict, test_config: Dict) -> str:
        """Generate enhanced JSON report with additional analysis."""
        # Add timestamp and additional metadata
        enhanced_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "SFT Local Model SQL Generation Test",
                "version": "2.0"
            },
            "test_config": test_config,
            "statistics": statistics,
            "detailed_analysis": self._perform_detailed_analysis(results, statistics),
            "results": results
        }
        
        json_path = self.output_dir / "enhanced_test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Enhanced JSON report saved to {json_path}")
        return str(json_path)
    
    def _perform_detailed_analysis(self, results: List[Dict], statistics: Dict) -> Dict:
        """Perform detailed analysis of the results."""
        analysis = {
            "performance_ranking": [],
            "rule_quality_analysis": {},
            "error_patterns": {},
            "database_performance": {}
        }
        
        # Performance ranking
        methods = [
            ("SFT Rules", "result_with_sft_rules"),
            ("Dataset Rules", "result_with_dataset_rules"),
            ("No Rules", "result_without_rules")
        ]
        
        for method_name, result_key in methods:
            correct_count = sum(1 for r in results if r[result_key] == 1)
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            analysis["performance_ranking"].append({
                "method": method_name,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count
            })
        
        # Sort by accuracy
        analysis["performance_ranking"].sort(key=lambda x: x["accuracy"], reverse=True)
        
        # Rule quality analysis
        sft_rules_samples = [r for r in results if r['generated_rules'].strip()]
        dataset_rules_samples = [r for r in results if r['dataset_rules'].strip()]
        
        analysis["rule_quality_analysis"] = {
            "sft_rules_coverage": len(sft_rules_samples) / len(results) if results else 0,
            "dataset_rules_coverage": len(dataset_rules_samples) / len(results) if results else 0,
            "average_sft_rule_length": np.mean([len(r['generated_rules']) for r in sft_rules_samples]) if sft_rules_samples else 0,
            "average_dataset_rule_length": np.mean([len(r['dataset_rules']) for r in dataset_rules_samples]) if dataset_rules_samples else 0
        }
        
        # Error patterns
        error_analysis = {
            "sft_rules_failures": 0,
            "dataset_rules_failures": 0,
            "no_rules_failures": 0,
            "common_failure_cases": []
        }
        
        for result in results:
            if result['result_with_sft_rules'] == -1:
                error_analysis["sft_rules_failures"] += 1
            if result['result_with_dataset_rules'] == -1:
                error_analysis["dataset_rules_failures"] += 1
            if result['result_without_rules'] == -1:
                error_analysis["no_rules_failures"] += 1
            
            # Identify cases where SFT rules failed but others succeeded
            if (result['result_with_sft_rules'] != 1 and 
                result['result_with_dataset_rules'] == 1 and 
                result['result_without_rules'] == 1):
                error_analysis["common_failure_cases"].append({
                    "sample_idx": result['sample_idx'],
                    "db_id": result['db_id'],
                    "question": result['question'][:100] + "..." if len(result['question']) > 100 else result['question']
                })
        
        analysis["error_patterns"] = error_analysis
        
        # Database performance analysis
        db_performance = {}
        for result in results:
            db_id = result['db_id']
            if db_id not in db_performance:
                db_performance[db_id] = {
                    "total_samples": 0,
                    "sft_correct": 0,
                    "dataset_correct": 0,
                    "no_rules_correct": 0
                }
            
            db_performance[db_id]["total_samples"] += 1
            if result['result_with_sft_rules'] == 1:
                db_performance[db_id]["sft_correct"] += 1
            if result['result_with_dataset_rules'] == 1:
                db_performance[db_id]["dataset_correct"] += 1
            if result['result_without_rules'] == 1:
                db_performance[db_id]["no_rules_correct"] += 1
        
        # Calculate accuracies for each database
        for db_id, perf in db_performance.items():
            total = perf["total_samples"]
            perf["sft_accuracy"] = perf["sft_correct"] / total if total > 0 else 0
            perf["dataset_accuracy"] = perf["dataset_correct"] / total if total > 0 else 0
            perf["no_rules_accuracy"] = perf["no_rules_correct"] / total if total > 0 else 0
        
        analysis["database_performance"] = db_performance
        
        return analysis
    
    def generate_summary_report(self, statistics: Dict, analysis: Dict) -> str:
        """Generate a human-readable summary report."""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SFT Local Model SQL Generation Test - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Samples Tested: {statistics['total_questions']}\n\n")
            
            # Method performance
            f.write("Method Performance:\n")
            for method in analysis["performance_ranking"]:
                f.write(f"  {method['method']}: {method['accuracy']:.3f} ({method['correct_count']}/{method['total_count']})\n")
            
            f.write(f"\nIMPROVEMENTS\n")
            f.write("-" * 12 + "\n")
            improvements = statistics['improvements']
            f.write(f"SFT Rules vs No Rules: {improvements['sft_vs_no_rules']['absolute']:+.3f} ({improvements['sft_vs_no_rules']['relative']:+.1f}%)\n")
            f.write(f"Dataset Rules vs No Rules: {improvements['dataset_vs_no_rules']['absolute']:+.3f} ({improvements['dataset_vs_no_rules']['relative']:+.1f}%)\n")
            f.write(f"SFT Rules vs Dataset Rules: {improvements['sft_vs_dataset']['absolute']:+.3f} ({improvements['sft_vs_dataset']['relative']:+.1f}%)\n")
            
            f.write(f"\nRULE QUALITY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            rule_analysis = analysis["rule_quality_analysis"]
            f.write(f"SFT Rules Coverage: {rule_analysis['sft_rules_coverage']:.3f}\n")
            f.write(f"Dataset Rules Coverage: {rule_analysis['dataset_rules_coverage']:.3f}\n")
            f.write(f"Average SFT Rule Length: {rule_analysis['average_sft_rule_length']:.1f} characters\n")
            f.write(f"Average Dataset Rule Length: {rule_analysis['average_dataset_rule_length']:.1f} characters\n")
            
            f.write(f"\nERROR PATTERNS\n")
            f.write("-" * 13 + "\n")
            error_patterns = analysis["error_patterns"]
            f.write(f"SFT Rules Failures: {error_patterns['sft_rules_failures']}\n")
            f.write(f"Dataset Rules Failures: {error_patterns['dataset_rules_failures']}\n")
            f.write(f"No Rules Failures: {error_patterns['no_rules_failures']}\n")
            f.write(f"Common Failure Cases: {len(error_patterns['common_failure_cases'])}\n")
            
            # Database performance summary
            f.write(f"\nDATABASE PERFORMANCE\n")
            f.write("-" * 19 + "\n")
            db_perf = analysis["database_performance"]
            for db_id, perf in sorted(db_perf.items()):
                f.write(f"{db_id}:\n")
                f.write(f"  SFT: {perf['sft_accuracy']:.3f}, Dataset: {perf['dataset_accuracy']:.3f}, No Rules: {perf['no_rules_accuracy']:.3f}\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
        return str(report_path)


class SFTResultsTester:
    """Main class for testing SFT results."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the SFT results tester."""
        self.args = args
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create file handler for logging
        log_file = self.output_dir / "sft_local_test.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear root logger handlers to prevent duplicate logs
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)  # Set to WARNING to avoid duplicate logs
        
        # Load rule generation model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            args.base_model, 
            args.adapter_path, 
            args.trust_remote_code
        )
        
        # Load SQL generation model and tokenizer
        self.sql_model, self.sql_tokenizer = load_sql_model_and_tokenizer(
            args.sql_model,
            args.trust_remote_code
        )
        
        # Initialize components
        self.sql_comparator = SQLTestComparator(args.db_root_path)
        self.schema_helper = SchemaInformation()
        
        # Load test samples
        self.test_samples = load_test_samples(args.dataset, args.num_samples)
        
        # Statistics for three test types
        self.total_questions = 0
        self.correct_answers_with_sft_rules = 0
        self.correct_answers_without_rules = 0
        self.correct_answers_with_dataset_rules = 0
        self.failed_executions_with_sft_rules = 0
        self.failed_executions_without_rules = 0
        self.failed_executions_with_dataset_rules = 0
        self.results = []
        
        # Initialize results analyzer
        self.analyzer = SFTResultsAnalyzer(str(self.output_dir))
        
        self.logger.info("SFT Results Tester initialized successfully")
    
    def test_single_sample(self, sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """Test a single sample through three pipelines (SFT rules, dataset rules, and without rules)."""
        question = sample.get("question", "").strip()
        ground_truth = sample.get("ground_truth", "").strip()
        db_id = sample.get("db_id", "").strip()
        dataset_rules = sample.get("rules", [])
        
        # Only log basic info, detailed output will be conditional
        self.logger.info(f"\nSample {sample_idx + 1}/{len(self.test_samples)} - {db_id}")
        
        result = {
            "sample_idx": sample_idx,
            "question": question,
            "ground_truth": ground_truth,
            "db_id": db_id,
            "generated_rules": "",
            "dataset_rules": "",
            "sql_with_sft_rules": "",
            "sql_without_rules": "",
            "sql_with_dataset_rules": "",
            "result_with_sft_rules": -1,  # -1: error, 0: incorrect, 1: correct
            "result_without_rules": -1,
            "result_with_dataset_rules": -1,
            "error_message": ""
        }
        
        try:
            # Get database schema
            schema = ""
            if db_id:
                db_path = os.path.join(self.args.db_root_path, db_id, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    try:
                        schema = self.schema_helper.generate_schema_info(db_path)
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
            
            # Step 1: Generate rules using the fine-tuned model
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
            rule_prompt = build_rule_generation_prompt(instruction, schema, question)
            
            generated_rules = generate_rules(
                self.model, 
                self.tokenizer, 
                rule_prompt, 
                self.args.max_new_tokens,
                self.args.temperature, 
                self.args.do_sample
            )
            
            result["generated_rules"] = generated_rules
            
            # Step 2: Extract dataset rules
            dataset_rules_text = extract_rules_from_dataset(dataset_rules)
            result["dataset_rules"] = dataset_rules_text
            
            # Step 3A: Test with SFT-generated rules
            sql_prompt = build_sql_generation_prompt_with_rules(question, schema, generated_rules)
            
            try:
                response = generate_sql(
                    self.sql_model,
                    self.sql_tokenizer,
                    sql_prompt,
                    self.args.sql_max_new_tokens,
                    self.args.sql_temperature,
                    self.args.sql_do_sample
                )
                predicted_sql_with_sft_rules = extract_sql_from_response(response)
                result["sql_with_sft_rules"] = predicted_sql_with_sft_rules
                
                # Verify the result with SFT rules
                verification_result_with_sft_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_with_sft_rules, ground_truth, db_id
                )
                result["result_with_sft_rules"] = verification_result_with_sft_rules
                
                if verification_result_with_sft_rules == 1:
                    self.correct_answers_with_sft_rules += 1
                else:
                    if verification_result_with_sft_rules == -1:
                        self.failed_executions_with_sft_rules += 1
                    
            except Exception as e:
                result["result_with_sft_rules"] = -1
                self.failed_executions_with_sft_rules += 1
            
            # Step 3B: Test with dataset rules
            sql_prompt = build_sql_generation_prompt_with_dataset_rules(question, schema, dataset_rules_text)
            
            try:
                response = generate_sql(
                    self.sql_model,
                    self.sql_tokenizer,
                    sql_prompt,
                    self.args.sql_max_new_tokens,
                    self.args.sql_temperature,
                    self.args.sql_do_sample
                )
                predicted_sql_with_dataset_rules = extract_sql_from_response(response)
                result["sql_with_dataset_rules"] = predicted_sql_with_dataset_rules
                
                # Verify the result with dataset rules
                verification_result_with_dataset_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_with_dataset_rules, ground_truth, db_id
                )
                result["result_with_dataset_rules"] = verification_result_with_dataset_rules
                
                if verification_result_with_dataset_rules == 1:
                    self.correct_answers_with_dataset_rules += 1
                else:
                    if verification_result_with_dataset_rules == -1:
                        self.failed_executions_with_dataset_rules += 1
                    
            except Exception as e:
                result["result_with_dataset_rules"] = -1
                self.failed_executions_with_dataset_rules += 1
            
            # Step 3C: Test without rules
            sql_prompt = build_sql_generation_prompt_without_rules(question, schema)
            
            try:
                response = generate_sql(
                    self.sql_model,
                    self.sql_tokenizer,
                    sql_prompt,
                    self.args.sql_max_new_tokens,
                    self.args.sql_temperature,
                    self.args.sql_do_sample
                )
                predicted_sql_without_rules = extract_sql_from_response(response)
                result["sql_without_rules"] = predicted_sql_without_rules
                
                # Verify the result without rules
                verification_result_without_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_without_rules, ground_truth, db_id
                )
                result["result_without_rules"] = verification_result_without_rules
                
                if verification_result_without_rules == 1:
                    self.correct_answers_without_rules += 1
                else:
                    if verification_result_without_rules == -1:
                        self.failed_executions_without_rules += 1
                    
            except Exception as e:
                result["result_without_rules"] = -1
                self.failed_executions_without_rules += 1
            
            # Update total questions count
            self.total_questions += 1
            
            # Check if fine-tuned model SQL is wrong while other two are correct
            sft_result = result["result_with_sft_rules"]
            dataset_result = result["result_with_dataset_rules"]
            no_rules_result = result["result_without_rules"]
            
            # Only output detailed information when SFT rules produce wrong SQL but other two produce correct SQL
            if (sft_result != 1 and dataset_result == 1 and no_rules_result == 1):
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"🔍 SPECIAL CASE: Fine-tuned model failed while others succeeded")
                self.logger.info(f"{'='*80}")
                self.logger.info(f"Sample {sample_idx + 1} - Database: {db_id}")
                self.logger.info(f"Question: {question}")
                self.logger.info(f"Ground Truth: {ground_truth}")
                
                self.logger.info(f"\n📋 EXTRACTED RULES (from dataset):")
                self.logger.info(f"{dataset_rules_text}")
                
                self.logger.info(f"\n🤖 GENERATED RULES (from fine-tuned model):")
                self.logger.info(f"{generated_rules}")
                
                self.logger.info(f"\n🔍 SQL COMPARISON:")
                self.logger.info(f"With SFT rules: {result['sql_with_sft_rules']}")
                self.logger.info(f"With dataset rules: {result['sql_with_dataset_rules']}")
                self.logger.info(f"Without rules: {result['sql_without_rules']}")
                
                self.logger.info(f"\n❌ ERROR ANALYSIS:")
                if sft_result == 0:
                    self.logger.info(f"Fine-tuned model SQL is incorrect (result: {sft_result})")
                elif sft_result == -1:
                    self.logger.info(f"Fine-tuned model SQL execution failed (result: {sft_result})")
                else:
                    self.logger.info(f"Fine-tuned model SQL has unexpected result: {sft_result}")
                
                self.logger.info(f"Dataset rules SQL is correct (result: {dataset_result})")
                self.logger.info(f"No rules SQL is correct (result: {no_rules_result})")
                self.logger.info(f"{'='*80}")
            else:
                # Simple output for normal cases
                self.logger.info(f"  SFT: {'✅' if sft_result == 1 else '❌'}, Dataset: {'✅' if dataset_result == 1 else '❌'}, No rules: {'✅' if no_rules_result == 1 else '❌'}")
            
            # Log current accuracies (simplified)
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions if self.total_questions > 0 else 0
            
            self.logger.info(f"  Accuracies: SFT={accuracy_with_sft_rules:.3f}, Dataset={accuracy_with_dataset_rules:.3f}, No rules={accuracy_without_rules:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing sample: {e}")
            result["error_message"] = str(e)
            self.failed_executions_with_sft_rules += 1
            self.failed_executions_with_dataset_rules += 1
            self.failed_executions_without_rules += 1
        
        return result
    
    def run_test(self):
        """Run the complete test on all samples."""
        self.logger.info(f"\n🧪 Starting SFT Results Test")
        self.logger.info(f"Testing {len(self.test_samples)} samples...")
        self.logger.info("=" * 80)
        
        for i, sample in enumerate(tqdm(self.test_samples, desc="Processing samples")):
            result = self.test_single_sample(sample, i)
            self.results.append(result)
        
        # Print final statistics
        self.print_final_statistics()
        
        # Save results
        self.save_results()
    
    def print_final_statistics(self):
        """Print final test statistics."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("SFT RESULTS TEST COMPLETED")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total Samples Processed: {self.total_questions}")
        
        if self.total_questions > 0:
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions
            
            sft_improvement = accuracy_with_sft_rules - accuracy_without_rules
            dataset_improvement = accuracy_with_dataset_rules - accuracy_without_rules
            sft_vs_dataset = accuracy_with_sft_rules - accuracy_with_dataset_rules
            
            self.logger.info(f"\n📊 FINAL RESULTS:")
            self.logger.info(f"  With SFT Rules:")
            self.logger.info(f"    Correct Answers: {self.correct_answers_with_sft_rules}")
            self.logger.info(f"    Failed Executions: {self.failed_executions_with_sft_rules}")
            self.logger.info(f"    Accuracy: {accuracy_with_sft_rules:.3f} ({accuracy_with_sft_rules*100:.1f}%)")
            
            self.logger.info(f"  With Dataset Rules:")
            self.logger.info(f"    Correct Answers: {self.correct_answers_with_dataset_rules}")
            self.logger.info(f"    Failed Executions: {self.failed_executions_with_dataset_rules}")
            self.logger.info(f"    Accuracy: {accuracy_with_dataset_rules:.3f} ({accuracy_with_dataset_rules*100:.1f}%)")
            
            self.logger.info(f"  Without Rules (Baseline):")
            self.logger.info(f"    Correct Answers: {self.correct_answers_without_rules}")
            self.logger.info(f"    Failed Executions: {self.failed_executions_without_rules}")
            self.logger.info(f"    Accuracy: {accuracy_without_rules:.3f} ({accuracy_without_rules*100:.1f}%)")
            
            self.logger.info(f"\n🎯 IMPROVEMENTS:")
            self.logger.info(f"  SFT Rules vs No Rules: {sft_improvement:+.3f} ({sft_improvement*100:+.1f}%)")
            self.logger.info(f"  Dataset Rules vs No Rules: {dataset_improvement:+.3f} ({dataset_improvement*100:+.1f}%)")
            self.logger.info(f"  SFT Rules vs Dataset Rules: {sft_vs_dataset:+.3f} ({sft_vs_dataset*100:+.1f}%)")
            
            self.logger.info(f"\n🏆 ANALYSIS:")
            if sft_improvement > 0:
                self.logger.info(f"    ✅ SFT rules help improve performance over baseline!")
            elif sft_improvement < 0:
                self.logger.info(f"    ❌ SFT rules hurt performance compared to baseline!")
            else:
                self.logger.info(f"    ➖ SFT rules have no effect on performance")
                
            if dataset_improvement > 0:
                self.logger.info(f"    ✅ Dataset rules help improve performance over baseline!")
            elif dataset_improvement < 0:
                self.logger.info(f"    ❌ Dataset rules hurt performance compared to baseline!")
            else:
                self.logger.info(f"    ➖ Dataset rules have no effect on performance")
                
            if sft_vs_dataset > 0:
                self.logger.info(f"    🎯 SFT rules perform better than dataset rules!")
            elif sft_vs_dataset < 0:
                self.logger.info(f"    📚 Dataset rules perform better than SFT rules!")
            else:
                self.logger.info(f"    ⚖️  SFT rules and dataset rules perform equally!")
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")
    
    def save_results(self):
        """Save test results using the comprehensive analyzer."""
        try:
            # Calculate statistics
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions if self.total_questions > 0 else 0
            
            sft_improvement = accuracy_with_sft_rules - accuracy_without_rules
            dataset_improvement = accuracy_with_dataset_rules - accuracy_without_rules
            sft_vs_dataset = accuracy_with_sft_rules - accuracy_with_dataset_rules
            
            # Prepare data structures
            test_config = {
                "base_model": self.args.base_model,
                "adapter_path": self.args.adapter_path,
                "sql_model": self.args.sql_model,
                "dataset": self.args.dataset,
                "num_samples": self.args.num_samples,
                "rule_generation_params": {
                    "max_new_tokens": self.args.max_new_tokens,
                    "temperature": self.args.temperature,
                    "do_sample": self.args.do_sample
                },
                "sql_generation_params": {
                    "max_new_tokens": self.args.sql_max_new_tokens,
                    "temperature": self.args.sql_temperature,
                    "do_sample": self.args.sql_do_sample
                }
            }
            
            statistics = {
                "total_questions": self.total_questions,
                "with_sft_rules": {
                    "correct_answers": self.correct_answers_with_sft_rules,
                    "failed_executions": self.failed_executions_with_sft_rules,
                    "accuracy": accuracy_with_sft_rules
                },
                "with_dataset_rules": {
                    "correct_answers": self.correct_answers_with_dataset_rules,
                    "failed_executions": self.failed_executions_with_dataset_rules,
                    "accuracy": accuracy_with_dataset_rules
                },
                "without_rules": {
                    "correct_answers": self.correct_answers_without_rules,
                    "failed_executions": self.failed_executions_without_rules,
                    "accuracy": accuracy_without_rules
                },
                "improvements": {
                    "sft_vs_no_rules": {
                        "absolute": sft_improvement,
                        "relative": sft_improvement * 100,
                        "description": f"SFT rules provide {sft_improvement:+.3f} ({sft_improvement*100:+.1f}%) accuracy improvement over baseline"
                    },
                    "dataset_vs_no_rules": {
                        "absolute": dataset_improvement,
                        "relative": dataset_improvement * 100,
                        "description": f"Dataset rules provide {dataset_improvement:+.3f} ({dataset_improvement*100:+.1f}%) accuracy improvement over baseline"
                    },
                    "sft_vs_dataset": {
                        "absolute": sft_vs_dataset,
                        "relative": sft_vs_dataset * 100,
                        "description": f"SFT rules provide {sft_vs_dataset:+.3f} ({sft_vs_dataset*100:+.1f}%) accuracy improvement over dataset rules"
                    }
                }
            }
            
            # Generate comprehensive reports
            self.logger.info("Generating comprehensive reports...")
            
            # 1. Generate CSV reports
            csv_path = self.analyzer.generate_csv_report(self.results, statistics)
            
            # 2. Generate visualizations
            self.analyzer.generate_visualizations(self.results, statistics)
            
            # 3. Generate enhanced JSON report
            json_path = self.analyzer.generate_enhanced_json_report(self.results, statistics, test_config)
            
            # 4. Perform detailed analysis and generate summary report
            detailed_analysis = self.analyzer._perform_detailed_analysis(self.results, statistics)
            summary_path = self.analyzer.generate_summary_report(statistics, detailed_analysis)
            
            self.logger.info("✅ All reports generated successfully!")
            self.logger.info(f"📁 Results directory: {self.output_dir}")
            self.logger.info(f"📊 Generated files:")
            self.logger.info(f"   - {csv_path} (detailed CSV)")
            self.logger.info(f"   - {self.output_dir}/summary_statistics.csv (summary CSV)")
            self.logger.info(f"   - {json_path} (enhanced JSON)")
            self.logger.info(f"   - {summary_path} (summary report)")
            self.logger.info(f"   - {self.output_dir}/sft_test_results_analysis.png (main visualization)")
            self.logger.info(f"   - {self.output_dir}/detailed_sample_comparison.png (detailed comparison)")
            self.logger.info(f"   - {self.output_dir}/sft_local_test.log (test log)")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


def main():
    """Main function to run the SFT results test."""
    logger = logging.getLogger(__name__)
    
    # Set up basic logging first (will be overridden by SFTResultsTester)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting SFT Results Test")
    logger.info("=" * 60)
    
    args = parse_args()
    
    # Validate paths
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"Base model path not found: {args.base_model}")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if not os.path.isdir(args.sql_model):
        raise FileNotFoundError(f"SQL model path not found: {args.sql_model}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if not os.path.isdir(args.db_root_path):
        raise FileNotFoundError(f"Database root path not found: {args.db_root_path}")
    
    try:
        # Initialize tester (this will set up proper logging)
        tester = SFTResultsTester(args)
        
        # Test local models
        logger.info("✅ Local models loaded successfully!")
        
        # Run test
        tester.run_test()
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
