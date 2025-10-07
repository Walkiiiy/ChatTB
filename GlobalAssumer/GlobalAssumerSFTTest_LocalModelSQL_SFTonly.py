#!/usr/bin/env python
"""
SFT Model Rules Testing Script - Optimized Version

This script tests SFT fine-tuned model generated rules for SQL generation in two phases:

Phase 1 - Rules Generation:
1. Load SFT model with LoRA adapter
2. Generate rules for all test samples
3. Save rules to intermediate file
4. Unload SFT model from memory

Phase 2 - SQL Testing:
1. Load SQL generation model
2. Generate SQL using SFT-generated rules
3. Validate SQL against ground truth
4. Generate comprehensive reports

Benefits:
- Memory efficient: Only one model loaded at a time
- Faster iteration: Can reuse generated rules for multiple SQL tests
- Clear separation of concerns: Rules generation vs SQL testing

Enhanced with comprehensive reporting:
- CSV reports with detailed results and summary statistics
- Visualization plots showing accuracy metrics
- Enhanced JSON reports with detailed analysis
- Human-readable summary reports

Usage:
    python GlobalAssumerSFTTest_LocalModelSQL_SFTonly.py \
        --base_model /path/to/base/model \
        --adapter_path /path/to/adapter \
        --sql_model /path/to/sql/model \
        --dataset /path/to/condensed_rules.json \
        --num_samples 10
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
                       help="Number of random samples to test (only samples with rules will be selected). Use -1 to test all samples.")
    parser.add_argument("--output_dir", type=str, default="sft_test_results",
                       help="Output directory for test results and reports")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=65535,
                       help="Maximum number of new tokens to generate for rules")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for rule generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during rule generation")
    
    # SQL generation parameters
    parser.add_argument("--sql_max_new_tokens", type=int, default=1024,
                       help="Maximum number of new tokens to generate for SQL")
    parser.add_argument("--sql_temperature", type=float, default=0.1,
                       help="Temperature for SQL generation")
    parser.add_argument("--sql_do_sample", action="store_true",
                       help="Enable sampling during SQL generation")
    
    return parser.parse_args()




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
    
    # Handle -1 as "test all samples"
    if num_samples == -1:
        sampled_items = items_with_rules
        logger.info(f"✅ Testing ALL {len(sampled_items)} samples with rules")
    else:
        # Sample random items from those with rules
        sampled_items = random.sample(items_with_rules, min(num_samples, len(items_with_rules)))
        logger.info(f"✅ Selected {len(sampled_items)} test samples (all with rules)")
    
    return sampled_items


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


class RulesGenerator:
    """Phase 1: Generate rules for all test samples using SFT model."""
    
    def __init__(self, base_model_path: str, adapter_path: str, trust_remote_code: bool, 
                 max_new_tokens: int, temperature: float, do_sample: bool):
        """Initialize the rules generator."""
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.schema_helper = SchemaInformation()
    
    def load_model(self):
        """Load the SFT model with adapter."""
        self.logger.info(f"📖 Loading base model from: {self.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            use_fast=True, 
            trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        
        # Load adapter
        self.logger.info(f"🔧 Loading adapter from: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        self.logger.info("✅ SFT model and adapter loaded successfully!")
    
    def unload_model(self):
        """Unload model to free memory."""
        self.logger.info("🗑️  Unloading SFT model from memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("✅ SFT model unloaded successfully!")
    
    def build_rule_generation_prompt(self, instruction: str, schema: str, question: str) -> str:
        """Build prompt for rule generation."""
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Database Schema:\n{schema}\n\n"
            f"Question:\n{question}\n\n"
            f"generated rules:\n"
        )
    
    def generate_rules_for_sample(self, question: str, schema: str) -> str:
        """Generate rules for a single sample."""
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
        
        prompt = self.build_rule_generation_prompt(instruction, schema, question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1 if self.do_sample else 1,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_all_rules(self, samples: List[Dict], db_root_path: str, output_path: str) -> List[Dict]:
        """Generate rules for all samples and save to file."""
        self.logger.info(f"🚀 Starting rules generation for {len(samples)} samples...")
        
        # Load model
        self.load_model()
        
        results = []
        
        for idx, sample in enumerate(tqdm(samples, desc="Generating rules")):
            question = sample.get("question", "").strip()
            db_id = sample.get("db_id", "").strip()
            ground_truth = sample.get("ground_truth", "").strip()
            
            try:
                # Get database schema
                schema = ""
                if db_id:
                    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
                    if os.path.exists(db_path):
                        schema = self.schema_helper.generate_schema_info(db_path)
                    else:
                        self.logger.warning(f"⚠️  Database file not found: {db_path}")
                        continue
                
                if not schema:
                    self.logger.warning(f"⚠️  No schema found for {db_id}")
                    continue
                
                # Generate rules
                generated_rules = self.generate_rules_for_sample(question, schema)
                
                results.append({
                    "sample_idx": idx,
                    "question": question,
                    "db_id": db_id,
                    "ground_truth": ground_truth,
                    "generated_rules": generated_rules,
                    "schema": schema
                })
                
            except Exception as e:
                self.logger.error(f"❌ Error generating rules for sample {idx}: {e}")
                results.append({
                    "sample_idx": idx,
                    "question": question,
                    "db_id": db_id,
                    "ground_truth": ground_truth,
                    "generated_rules": "",
                    "schema": "",
                    "error": str(e)
                })
        
        # Save results to file
        self.logger.info(f"💾 Saving generated rules to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Generated rules for {len(results)} samples")
        
        # Unload model
        self.unload_model()
        
        return results


class SQLTester:
    """Phase 2: Test SQL generation using SFT-generated rules."""
    
    def __init__(self, sql_model_path: str, trust_remote_code: bool, db_root_path: str,
                 max_new_tokens: int, temperature: float, do_sample: bool):
        """Initialize the SQL tester."""
        self.sql_model_path = sql_model_path
        self.trust_remote_code = trust_remote_code
        self.db_root_path = db_root_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.logger = logging.getLogger(__name__)
        self.sql_model = None
        self.sql_tokenizer = None
        self.sql_comparator = SQLTestComparator(db_root_path)
        
        # Statistics
        self.total_samples = 0
        self.correct_answers = 0
        self.failed_executions = 0
    
    def load_model(self):
        """Load the SQL generation model."""
        self.logger.info(f"📖 Loading SQL model from: {self.sql_model_path}")
        
        # Load tokenizer
        self.sql_tokenizer = AutoTokenizer.from_pretrained(
            self.sql_model_path, 
            use_fast=True, 
            trust_remote_code=self.trust_remote_code
        )
        if self.sql_tokenizer.pad_token is None:
            self.sql_tokenizer.pad_token = self.sql_tokenizer.eos_token
        self.sql_tokenizer.padding_side = "right"
        
        # Load SQL model
        self.sql_model = AutoModelForCausalLM.from_pretrained(
            self.sql_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        
        self.logger.info("✅ SQL generation model loaded successfully!")
    
    def unload_model(self):
        """Unload model to free memory."""
        self.logger.info("🗑️  Unloading SQL model from memory...")
        del self.sql_model
        del self.sql_tokenizer
        self.sql_model = None
        self.sql_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("✅ SQL model unloaded successfully!")
    
    def build_sql_generation_prompt(self, question: str, schema: str, rules: str) -> str:
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
    
    def generate_sql(self, question: str, schema: str, rules: str) -> str:
        """Generate SQL for a sample."""
        prompt = self.build_sql_generation_prompt(question, schema, rules)
        
        # Tokenize input
        inputs = self.sql_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs["input_ids"].to(self.sql_model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.sql_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.sql_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.sql_tokenizer.pad_token_id,
                eos_token_id=self.sql_tokenizer.eos_token_id,
                num_beams=1 if self.do_sample else 1,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.sql_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return extract_sql_from_response(response.strip())
    
    def test_all_samples(self, rules_data: List[Dict], output_dir: str) -> List[Dict]:
        """Test SQL generation for all samples with generated rules."""
        self.logger.info(f"🚀 Starting SQL testing for {len(rules_data)} samples...")
        
        # Load model
        self.load_model()
        
        results = []
        
        for sample in tqdm(rules_data, desc="Testing SQL generation"):
            sample_idx = sample.get("sample_idx", 0)
            question = sample.get("question", "")
            db_id = sample.get("db_id", "")
            ground_truth = sample.get("ground_truth", "")
            generated_rules = sample.get("generated_rules", "")
            schema = sample.get("schema", "")
            
            result = {
                "sample_idx": sample_idx,
                "question": question,
                "db_id": db_id,
                "ground_truth": ground_truth,
                "generated_rules": generated_rules,
                "generated_sql": "",
                "result": -1,  # -1: error, 0: incorrect, 1: correct
                "error_message": ""
            }
            
            try:
                # Skip if there was an error in rules generation
                if "error" in sample:
                    result["error_message"] = sample["error"]
                    self.failed_executions += 1
                    results.append(result)
                    continue
                
                # Generate SQL
                generated_sql = self.generate_sql(question, schema, generated_rules)
                result["generated_sql"] = generated_sql
                
                # Verify the result
                verification_result = self.sql_comparator.test_sql_with_db_id(
                    generated_sql, ground_truth, db_id
                )
                result["result"] = verification_result
                
                if verification_result == 1:
                    self.correct_answers += 1
                elif verification_result == -1:
                    self.failed_executions += 1
                
                self.total_samples += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error testing sample {sample_idx}: {e}")
                result["error_message"] = str(e)
                result["result"] = -1
                self.failed_executions += 1
                self.total_samples += 1
            
            results.append(result)
        
        self.logger.info(f"✅ Tested {len(results)} samples")
        
        # Unload model
        self.unload_model()
        
        return results


class ResultsAnalyzer:
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
                'generated_rules': result['generated_rules'],
                'generated_sql': result['generated_sql'],
                'result': result['result'],
                'error_message': result.get('error_message', '')
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Generate summary statistics CSV
        summary_data = {
            'Metric': [
                'Total Samples',
                'Correct Answers',
                'Incorrect Answers',
                'Failed Executions',
                'Accuracy'
            ],
            'Value': [
                statistics['total_samples'],
                statistics['correct_answers'],
                statistics['incorrect_answers'],
                statistics['failed_executions'],
                f"{statistics['accuracy']:.3f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"CSV reports saved to {csv_path} and {summary_csv_path}")
        return str(csv_path)
    
    def generate_visualizations(self, results: List[Dict], statistics: Dict) -> None:
        """Generate visualization plots for SFT rules testing results."""
        plt.style.use('default')
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SFT Model Rules Testing Results', fontsize=16, fontweight='bold')
        
        # 1. Overall Performance
        ax1 = axes[0, 0]
        categories = ['Correct', 'Incorrect', 'Failed']
        counts = [
            statistics['correct_answers'],
            statistics['incorrect_answers'],
            statistics['failed_executions']
        ]
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Yellow, Red
        bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
        ax1.set_title('Overall Performance')
        ax1.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Accuracy Metric
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, f"{statistics['accuracy']:.1%}", 
                ha='center', va='center', fontsize=72, fontweight='bold', color='#2E8B57')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Overall Accuracy', fontsize=14, pad=20)
        
        # 3. Result Distribution Pie Chart
        ax3 = axes[1, 0]
        sizes = [statistics['correct_answers'], statistics['incorrect_answers'], statistics['failed_executions']]
        labels = ['Correct', 'Incorrect', 'Failed']
        colors_pie = ['#2E8B57', '#FFD700', '#DC143C']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Result Distribution')
        
        # 4. Sample Results Heatmap (limited to first 20 samples)
        ax4 = axes[1, 1]
        sample_results = [r['result'] for r in results[:20]]
        sample_indices = [r['sample_idx'] for r in results[:20]]
        
        # Create heatmap data
        heatmap_data = np.array(sample_results).reshape(-1, 1)
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        colors_heat = ['red', 'yellow', 'green']
        cmap = ListedColormap(colors_heat)
        
        im = ax4.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        ax4.set_yticks(range(len(sample_indices)))
        ax4.set_yticklabels(sample_indices)
        ax4.set_xticks([])
        ax4.set_title(f'Sample Results (First {len(sample_results)})')
        ax4.set_ylabel('Sample Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, ticks=[-1, 0, 1])
        cbar.set_ticklabels(['Failed', 'Incorrect', 'Correct'])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "sft_test_results_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {plot_path}")
    
    
    def generate_json_report(self, results: List[Dict], statistics: Dict, test_config: Dict) -> str:
        """Generate JSON report with test results and statistics."""
        # Add timestamp and metadata
        report_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "SFT Model Rules Testing",
                "version": "3.0 - Optimized"
            },
            "test_config": test_config,
            "statistics": statistics,
            "results": results
        }
        
        json_path = self.output_dir / "test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report saved to {json_path}")
        return str(json_path)

    def generate_summary_report(self, statistics: Dict, test_config: Dict) -> str:
        """Generate a human-readable summary report."""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SFT Model Rules Testing - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TEST CONFIGURATION\n")
            f.write("-" * 18 + "\n")
            f.write(f"Base Model: {test_config['base_model']}\n")
            f.write(f"Adapter Path: {test_config['adapter_path']}\n")
            f.write(f"SQL Model: {test_config['sql_model']}\n")
            f.write(f"Dataset: {test_config['dataset']}\n")
            f.write(f"Num Samples: {test_config['num_samples']}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 19 + "\n")
            f.write(f"Total Samples Tested: {statistics['total_samples']}\n")
            f.write(f"Correct Answers: {statistics['correct_answers']} ({statistics['accuracy']:.1%})\n")
            f.write(f"Incorrect Answers: {statistics['incorrect_answers']}\n")
            f.write(f"Failed Executions: {statistics['failed_executions']}\n")
            f.write(f"Overall Accuracy: {statistics['accuracy']:.3f}\n\n")
            
            f.write("ANALYSIS\n")
            f.write("-" * 8 + "\n")
            if statistics['accuracy'] >= 0.8:
                f.write("✅ Excellent performance! SFT model generates high-quality rules.\n")
            elif statistics['accuracy'] >= 0.6:
                f.write("✓ Good performance. Rules are generally helpful.\n")
            elif statistics['accuracy'] >= 0.4:
                f.write("⚠ Moderate performance. Room for improvement.\n")
            else:
                f.write("❌ Low performance. Model may need additional training.\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
        return str(report_path)


def main():
    """Main function to run SFT model rules testing."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting SFT Model Rules Testing (Optimized Version)")
    logger.info("=" * 60)
    
    # Parse arguments
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging
    log_file = output_dir / "sft_test.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # ==================================================
        # PHASE 1: Generate Rules with SFT Model
        # ==================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: GENERATING RULES WITH SFT MODEL")
        logger.info("="*60)
        
        # Load test samples
        test_samples = load_test_samples(args.dataset, args.num_samples)
        
        if not test_samples:
            logger.error("❌ No test samples loaded. Exiting...")
            return
        # rules generator 传入base model和adaptor model，初始化不需要传入数据，是一个rule生成器
        # RuleGnenrator .generate_all_rules 传入samples，db_root_path，output_path,返回rules_data
        # rules_data 是一个list，每个元素是一个数据集中的dict
        # Initialize rules generator
        rules_generator = RulesGenerator(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
        
        # Generate rules for all samples
        rules_output_path = output_dir / "generated_rules.json"
        rules_data = rules_generator.generate_all_rules(
            samples=test_samples,
            db_root_path=args.db_root_path,
            output_path=str(rules_output_path)
        )
        
        logger.info(f"✅ Phase 1 completed! Rules saved to: {rules_output_path}")
        
        # ==================================================
        # PHASE 2: Test SQL Generation with Generated Rules
        # ==================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: TESTING SQL GENERATION WITH GENERATED RULES")
        logger.info("="*60)
        
        # Initialize SQL tester
        sql_tester = SQLTester(
            sql_model_path=args.sql_model,
            trust_remote_code=args.trust_remote_code,
            db_root_path=args.db_root_path,
            max_new_tokens=args.sql_max_new_tokens,
            temperature=args.sql_temperature,
            do_sample=args.sql_do_sample
        )
        
        # Test SQL generation
        test_results = sql_tester.test_all_samples(
            rules_data=rules_data,
            output_dir=str(output_dir)
        )
        
        logger.info(f"✅ Phase 2 completed!")
        
        # ==================================================
        # PHASE 3: Generate Reports and Analysis
        # ==================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: GENERATING REPORTS AND ANALYSIS")
        logger.info("="*60)
        
        # Calculate statistics
        accuracy = sql_tester.correct_answers / sql_tester.total_samples if sql_tester.total_samples > 0 else 0
        incorrect_answers = sql_tester.total_samples - sql_tester.correct_answers - sql_tester.failed_executions
        
        statistics = {
            "total_samples": sql_tester.total_samples,
            "correct_answers": sql_tester.correct_answers,
            "incorrect_answers": incorrect_answers,
            "failed_executions": sql_tester.failed_executions,
            "accuracy": accuracy
        }
        
        test_config = {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "sql_model": args.sql_model,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "rule_generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample
            },
            "sql_generation_params": {
                "max_new_tokens": args.sql_max_new_tokens,
                "temperature": args.sql_temperature,
                "do_sample": args.sql_do_sample
            }
        }
        
        # Initialize results analyzer
        analyzer = ResultsAnalyzer(str(output_dir))
        
        # Generate reports
        logger.info("📊 Generating CSV reports...")
        csv_path = analyzer.generate_csv_report(test_results, statistics)
        
        logger.info("📈 Generating visualizations...")
        analyzer.generate_visualizations(test_results, statistics)
        
        logger.info("📄 Generating JSON report...")
        json_path = analyzer.generate_json_report(test_results, statistics, test_config)
        
        logger.info("📝 Generating summary report...")
        summary_path = analyzer.generate_summary_report(statistics, test_config)
        
        # ==================================================
        # FINAL SUMMARY
        # ==================================================
        logger.info("\n" + "="*60)
        logger.info("✅ SFT MODEL RULES TESTING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   Total Samples: {statistics['total_samples']}")
        logger.info(f"   Correct Answers: {statistics['correct_answers']}")
        logger.info(f"   Incorrect Answers: {statistics['incorrect_answers']}")
        logger.info(f"   Failed Executions: {statistics['failed_executions']}")
        logger.info(f"   Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info(f"📊 Generated files:")
        logger.info(f"   - {rules_output_path.name} (generated rules)")
        logger.info(f"   - {Path(csv_path).name} (detailed results)")
        logger.info(f"   - summary_statistics.csv (summary statistics)")
        logger.info(f"   - {Path(json_path).name} (JSON report)")
        logger.info(f"   - {Path(summary_path).name} (summary report)")
        logger.info(f"   - sft_test_results_analysis.png (visualization)")
        logger.info(f"   - {log_file.name} (test log)")
        
        if accuracy >= 0.8:
            logger.info("\n🎉 Excellent performance!")
        elif accuracy >= 0.6:
            logger.info("\n✓ Good performance!")
        elif accuracy >= 0.4:
            logger.info("\n⚠ Moderate performance.")
        else:
            logger.info("\n❌ Low performance. Consider additional training.")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

