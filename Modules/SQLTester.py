'''

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
'''
from typing import List, Dict
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation
import logging


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