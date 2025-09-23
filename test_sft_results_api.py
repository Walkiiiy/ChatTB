#!/usr/bin/env python
"""
SFT Results Test Script - Triple Comparison Version

This script tests the SFT fine-tuned model by comparing THREE approaches:
1. Loading the fine-tuned globalAssumer model
2. Randomly selecting samples from a dataset (supports rules_res_type.json format)
3. Using the model to generate rules for each question
4. Testing THREE approaches:
   A. Generate SQL using DeepSeek WITH SFT-generated rules
   B. Generate SQL using DeepSeek WITH dataset definitional rules (ground truth)
   C. Generate SQL using DeepSeek WITHOUT rules (baseline)
5. Validating all three generated SQLs against ground truth
6. Comparing performance to measure the impact of different rule sources

This allows you to evaluate:
- Whether SFT model's generated rules improve SQL generation performance
- How SFT-generated rules compare to ground truth rules
- The overall effectiveness of rules-based SQL generation

Usage:
    python test_sft_results_api.py --base_model /path/to/base/model --adapter_path /path/to/adapter --dataset /path/to/rules_res_type.json --num_samples 10
"""

import argparse
import json
import random
import os
import logging
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import our custom classes
from Process_model.DeepSeekLLMClient import DeepSeekLLMClient
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation

from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SFT results testing."""
    parser = argparse.ArgumentParser(description="Test SFT fine-tuned model results")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model directory")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to the fine-tuned adapter directory")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of custom modeling code")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True,
                       help="Root directory containing database files")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of random samples to test")
    parser.add_argument("--output_file", type=str, default="sft_test_results.json",
                       help="Output file for test results")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate for rules")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for rule generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling during rule generation")
    
    # DeepSeek configuration
    parser.add_argument("--deepseek_model", type=str, default="deepseek-coder",
                       help="DeepSeek model name for SQL generation")
    parser.add_argument("--deepseek_api_key", type=str, default=None,
                       help="DeepSeek API key (if None, will try to get from environment)")
    
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


def load_test_samples(dataset_path: str, num_samples: int) -> List[Dict]:
    """Load random samples from the dataset for testing."""
    print(f"📋 Loading test samples from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to list if it's a dict
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    
    # Sample random items
    sampled_items = random.sample(items, min(num_samples, len(items)))
    
    print(f"✅ Selected {len(sampled_items)} test samples")
    return sampled_items


def extract_definitional_rules_from_dataset(rules: List[Dict]) -> str:
    """Extract definitional rules from dataset rules list."""
    definitional_rules = []
    for rule in rules:
        if str(rule.get("type", "")).lower() in ["definitional", "definational"]:
            condition = rule.get("condition", "").strip()
            operation = rule.get("operation", "").strip()
            if condition and operation:
                definitional_rules.append(f"Condition: {condition}\nOperation: {operation}")
    
    return "\n\n".join(definitional_rules) if definitional_rules else "No definitional rules found."


def build_rule_generation_prompt(instruction: str, schema: str, question: str) -> str:
    """Build prompt for rule generation."""
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"definitional rules:\n"
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


def build_sql_generation_prompt_with_rules(question: str, schema: str, rules: str) -> tuple:
    """Build system and user prompts for SQL generation with rules."""
    system_prompt = """
    You are a helpful assistant that writes valid SQLite queries.
    """
    
    user_prompt = f"""
    You will be given database schema, a question related to the database and some rules.
    You should generate a SQLite query that solves the question with the help of rules.
    The rules contain all the rules you should obey while generating the target SQL, you have to obey all of them.
    
    Database Schema:
    {schema}
    
    Question: {question}
    
    Rules: {rules}
    
    Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.
    """
    
    return system_prompt, user_prompt


def build_sql_generation_prompt_without_rules(question: str, schema: str) -> tuple:
    """Build system and user prompts for SQL generation without rules."""
    system_prompt = """
    You are a helpful assistant that writes valid SQLite queries.
    """
    
    user_prompt = f"""
    You will be given database schema and a question related to the database.
    You should generate a SQLite query that solves the question.
    
    Database Schema:
    {schema}
    
    Question: {question}
    
    Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.
    """
    
    return system_prompt, user_prompt


def build_sql_generation_prompt_with_dataset_rules(question: str, schema: str, dataset_rules: str) -> tuple:
    """Build system and user prompts for SQL generation with dataset rules."""
    system_prompt = """
    You are a helpful assistant that writes valid SQLite queries.
    """
    
    user_prompt = f"""
    You will be given database schema, a question related to the database and some rules.
    You should generate a SQLite query that solves the question with the help of rules.
    The rules contain all the rules you should obey while generating the target SQL, you have to obey all of them.
    
    Database Schema:
    {schema}
    
    Question: {question}
    
    Rules: {dataset_rules}
    
    Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.
    """
    
    return system_prompt, user_prompt


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


class SFTResultsTester:
    """Main class for testing SFT results."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the SFT results tester."""
        self.args = args
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            args.base_model, 
            args.adapter_path, 
            args.trust_remote_code
        )
        
        # Initialize DeepSeek client
        api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass --deepseek_api_key parameter.")
        
        self.llm_client = DeepSeekLLMClient(api_key=api_key, model=args.deepseek_model)
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
        
        self.logger.info("SFT Results Tester initialized successfully")
    
    def test_single_sample(self, sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """Test a single sample through three pipelines (SFT rules, dataset rules, and without rules)."""
        question = sample.get("question", "").strip()
        ground_truth = sample.get("ground_truth", "").strip()
        db_id = sample.get("db_id", "").strip()
        dataset_rules = sample.get("rules", [])
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{len(self.test_samples)}")
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
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
                        print(f"❌ Failed to generate schema for {db_id}: {e}")
                        result["error_message"] = f"Schema generation failed: {e}"
                        return result
                else:
                    print(f"❌ Database file not found: {db_path}")
                    result["error_message"] = f"Database file not found: {db_path}"
                    return result
            
            if not schema:
                print("❌ No schema found for this database")
                result["error_message"] = "No schema found"
                return result
            
            # Step 1: Generate rules using the fine-tuned model
            print(f"\n🤖 Step 1: Generating rules with fine-tuned model...")
            instruction = "Read the question and output only the definitional rules that apply."
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
            print(f"Generated Rules:\n{generated_rules}")
            
            # Step 2: Extract dataset rules
            print(f"\n📋 Step 2: Extracting dataset definitional rules...")
            dataset_definitional_rules = extract_definitional_rules_from_dataset(dataset_rules)
            result["dataset_rules"] = dataset_definitional_rules
            print(f"Dataset Rules:\n{dataset_definitional_rules}")
            
            # Step 3A: Test with SFT-generated rules
            print(f"\n🔍 Step 3A: Generating SQL with DeepSeek (WITH SFT rules)...")
            system_prompt, user_prompt = build_sql_generation_prompt_with_rules(question, schema, generated_rules)
            
            try:
                response = self.llm_client.chat(user_prompt, system_prompt)
                predicted_sql_with_sft_rules = extract_sql_from_response(response)
                result["sql_with_sft_rules"] = predicted_sql_with_sft_rules
                print(f"Generated SQL (with SFT rules): {predicted_sql_with_sft_rules}")
                
                # Verify the result with SFT rules
                verification_result_with_sft_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_with_sft_rules, ground_truth, db_id
                )
                result["result_with_sft_rules"] = verification_result_with_sft_rules
                
                if verification_result_with_sft_rules == 1:
                    self.correct_answers_with_sft_rules += 1
                    print("✅ Correct (with SFT rules)!")
                elif verification_result_with_sft_rules == 0:
                    print("❌ Incorrect result (with SFT rules)")
                else:
                    print("❌ Execution error (with SFT rules)")
                    self.failed_executions_with_sft_rules += 1
                    
            except Exception as e:
                print(f"❌ Error generating SQL with SFT rules: {e}")
                result["result_with_sft_rules"] = -1
                self.failed_executions_with_sft_rules += 1
            
            # Step 3B: Test with dataset rules
            print(f"\n🔍 Step 3B: Generating SQL with DeepSeek (WITH dataset rules)...")
            system_prompt, user_prompt = build_sql_generation_prompt_with_dataset_rules(question, schema, dataset_definitional_rules)
            
            try:
                response = self.llm_client.chat(user_prompt, system_prompt)
                predicted_sql_with_dataset_rules = extract_sql_from_response(response)
                result["sql_with_dataset_rules"] = predicted_sql_with_dataset_rules
                print(f"Generated SQL (with dataset rules): {predicted_sql_with_dataset_rules}")
                
                # Verify the result with dataset rules
                verification_result_with_dataset_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_with_dataset_rules, ground_truth, db_id
                )
                result["result_with_dataset_rules"] = verification_result_with_dataset_rules
                
                if verification_result_with_dataset_rules == 1:
                    self.correct_answers_with_dataset_rules += 1
                    print("✅ Correct (with dataset rules)!")
                elif verification_result_with_dataset_rules == 0:
                    print("❌ Incorrect result (with dataset rules)")
                else:
                    print("❌ Execution error (with dataset rules)")
                    self.failed_executions_with_dataset_rules += 1
                    
            except Exception as e:
                print(f"❌ Error generating SQL with dataset rules: {e}")
                result["result_with_dataset_rules"] = -1
                self.failed_executions_with_dataset_rules += 1
            
            # Step 3C: Test without rules
            print(f"\n🔍 Step 3C: Generating SQL with DeepSeek (WITHOUT rules)...")
            system_prompt, user_prompt = build_sql_generation_prompt_without_rules(question, schema)
            
            try:
                response = self.llm_client.chat(user_prompt, system_prompt)
                predicted_sql_without_rules = extract_sql_from_response(response)
                result["sql_without_rules"] = predicted_sql_without_rules
                print(f"Generated SQL (without rules): {predicted_sql_without_rules}")
                
                # Verify the result without rules
                verification_result_without_rules = self.sql_comparator.test_sql_with_db_id(
                    predicted_sql_without_rules, ground_truth, db_id
                )
                result["result_without_rules"] = verification_result_without_rules
                
                if verification_result_without_rules == 1:
                    self.correct_answers_without_rules += 1
                    print("✅ Correct (without rules)!")
                elif verification_result_without_rules == 0:
                    print("❌ Incorrect result (without rules)")
                else:
                    print("❌ Execution error (without rules)")
                    self.failed_executions_without_rules += 1
                    
            except Exception as e:
                print(f"❌ Error generating SQL without rules: {e}")
                result["result_without_rules"] = -1
                self.failed_executions_without_rules += 1
            
            # Update total questions count
            self.total_questions += 1
            
            # Print current accuracies
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions if self.total_questions > 0 else 0
            
            print(f"\n📊 Current Accuracies:")
            print(f"  With SFT rules: {accuracy_with_sft_rules:.3f} ({self.correct_answers_with_sft_rules}/{self.total_questions})")
            print(f"  With dataset rules: {accuracy_with_dataset_rules:.3f} ({self.correct_answers_with_dataset_rules}/{self.total_questions})")
            print(f"  Without rules: {accuracy_without_rules:.3f} ({self.correct_answers_without_rules}/{self.total_questions})")
            print(f"  SFT vs No rules: {accuracy_with_sft_rules - accuracy_without_rules:+.3f}")
            print(f"  Dataset vs No rules: {accuracy_with_dataset_rules - accuracy_without_rules:+.3f}")
            print(f"  SFT vs Dataset: {accuracy_with_sft_rules - accuracy_with_dataset_rules:+.3f}")
            
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            result["error_message"] = str(e)
            self.failed_executions_with_sft_rules += 1
            self.failed_executions_with_dataset_rules += 1
            self.failed_executions_without_rules += 1
        
        return result
    
    def run_test(self):
        """Run the complete test on all samples."""
        print(f"\n🧪 Starting SFT Results Test")
        print(f"Testing {len(self.test_samples)} samples...")
        print("=" * 80)
        
        for i, sample in enumerate(tqdm(self.test_samples, desc="Processing samples")):
            result = self.test_single_sample(sample, i)
            self.results.append(result)
        
        # Print final statistics
        self.print_final_statistics()
        
        # Save results
        self.save_results()
    
    def print_final_statistics(self):
        """Print final test statistics."""
        print(f"\n{'='*80}")
        print("SFT RESULTS TEST COMPLETED")
        print(f"{'='*80}")
        print(f"Total Samples Processed: {self.total_questions}")
        
        if self.total_questions > 0:
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions
            
            sft_improvement = accuracy_with_sft_rules - accuracy_without_rules
            dataset_improvement = accuracy_with_dataset_rules - accuracy_without_rules
            sft_vs_dataset = accuracy_with_sft_rules - accuracy_with_dataset_rules
            
            print(f"\n📊 FINAL RESULTS:")
            print(f"  With SFT Rules:")
            print(f"    Correct Answers: {self.correct_answers_with_sft_rules}")
            print(f"    Failed Executions: {self.failed_executions_with_sft_rules}")
            print(f"    Accuracy: {accuracy_with_sft_rules:.3f} ({accuracy_with_sft_rules*100:.1f}%)")
            
            print(f"  With Dataset Rules:")
            print(f"    Correct Answers: {self.correct_answers_with_dataset_rules}")
            print(f"    Failed Executions: {self.failed_executions_with_dataset_rules}")
            print(f"    Accuracy: {accuracy_with_dataset_rules:.3f} ({accuracy_with_dataset_rules*100:.1f}%)")
            
            print(f"  Without Rules (Baseline):")
            print(f"    Correct Answers: {self.correct_answers_without_rules}")
            print(f"    Failed Executions: {self.failed_executions_without_rules}")
            print(f"    Accuracy: {accuracy_without_rules:.3f} ({accuracy_without_rules*100:.1f}%)")
            
            print(f"\n🎯 IMPROVEMENTS:")
            print(f"  SFT Rules vs No Rules: {sft_improvement:+.3f} ({sft_improvement*100:+.1f}%)")
            print(f"  Dataset Rules vs No Rules: {dataset_improvement:+.3f} ({dataset_improvement*100:+.1f}%)")
            print(f"  SFT Rules vs Dataset Rules: {sft_vs_dataset:+.3f} ({sft_vs_dataset*100:+.1f}%)")
            
            print(f"\n🏆 ANALYSIS:")
            if sft_improvement > 0:
                print(f"    ✅ SFT rules help improve performance over baseline!")
            elif sft_improvement < 0:
                print(f"    ❌ SFT rules hurt performance compared to baseline!")
            else:
                print(f"    ➖ SFT rules have no effect on performance")
                
            if dataset_improvement > 0:
                print(f"    ✅ Dataset rules help improve performance over baseline!")
            elif dataset_improvement < 0:
                print(f"    ❌ Dataset rules hurt performance compared to baseline!")
            else:
                print(f"    ➖ Dataset rules have no effect on performance")
                
            if sft_vs_dataset > 0:
                print(f"    🎯 SFT rules perform better than dataset rules!")
            elif sft_vs_dataset < 0:
                print(f"    📚 Dataset rules perform better than SFT rules!")
            else:
                print(f"    ⚖️  SFT rules and dataset rules perform equally!")
        
        print(f"\nResults saved to: {self.args.output_file}")
    
    def save_results(self):
        """Save test results to JSON file."""
        try:
            accuracy_with_sft_rules = self.correct_answers_with_sft_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_with_dataset_rules = self.correct_answers_with_dataset_rules / self.total_questions if self.total_questions > 0 else 0
            accuracy_without_rules = self.correct_answers_without_rules / self.total_questions if self.total_questions > 0 else 0
            
            sft_improvement = accuracy_with_sft_rules - accuracy_without_rules
            dataset_improvement = accuracy_with_dataset_rules - accuracy_without_rules
            sft_vs_dataset = accuracy_with_sft_rules - accuracy_with_dataset_rules
            
            output_data = {
                "test_config": {
                    "base_model": self.args.base_model,
                    "adapter_path": self.args.adapter_path,
                    "dataset": self.args.dataset,
                    "num_samples": self.args.num_samples,
                    "deepseek_model": self.args.deepseek_model,
                    "generation_params": {
                        "max_new_tokens": self.args.max_new_tokens,
                        "temperature": self.args.temperature,
                        "do_sample": self.args.do_sample
                    }
                },
                "statistics": {
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
                },
                "results": self.results
            }
            
            with open(self.args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {self.args.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


def main():
    """Main function to run the SFT results test."""
    print("🚀 Starting SFT Results Test")
    print("=" * 60)
    
    args = parse_args()
    
    # Validate paths
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"Base model path not found: {args.base_model}")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if not os.path.isdir(args.db_root_path):
        raise FileNotFoundError(f"Database root path not found: {args.db_root_path}")
    
    try:
        # Initialize tester
        tester = SFTResultsTester(args)
        
        # Test API connection first
        print("Testing DeepSeek API connection...")
        if not tester.llm_client.test_connection():
            print("❌ Failed to connect to DeepSeek API. Please check your API key and internet connection.")
            return
        print("✅ DeepSeek API connection successful!")
        
        # Run test
        tester.run_test()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
