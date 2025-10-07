'''
# rules generator 传入base model和adaptor model，初始化不需要传入数据，是一个rule生成器
# RuleGnenrator .generate_all_rules 传入samples，db_root_path，output_path,返回rules_data
# rules_data 是一个list，每个元素是一个数据集中的dict
# RuleGenerator.generate_rules_for_sample 传入question，schema，返回generated_rules
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
'''
from typing import List, Dict
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from Process_model.SchemaInformation import SchemaInformation
import logging

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