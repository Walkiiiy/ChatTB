'''
generating input data for global assumer
'''
import json
from typing import List, Dict, Iterable
import os
from transformers import AutoTokenizer
from datasets import Dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Any
import torch
class GlobalAssumerDataloader:
    def __init__(self, tokenizer: AutoTokenizer, rules_json_path: str,schema_path: str, skip_no_rules: bool=True, batch_size: int=1,max_prompt_length: int=None,schema_rows: int=0):
        self.max_prompt_length = max_prompt_length
        self.rules_json_path = rules_json_path
        self.skip_no_rules = skip_no_rules
        self.schema_rows = schema_rows
        self.schema_path=schema_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
    def __build_io_pair(self,schema: str, question: str, rules: List[str]) -> str:
        """
        Build a training prompt from instruction, schema, question, and rules.
        
        Args:
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

    def __iter_rules_items(self,rules_json_path: str) -> Iterable[Dict]:
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
    
    def __get_prompts_from_rules(self,tokenizer: AutoTokenizer,max_prompt_length: int, rules_json_path: str, schema_path: str, skip_no_rules: bool, schema_rows: int) -> List[Dict[str, str]]:
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
        # schema_helper = SchemaInformation()
        
        print(f"Loading rules from: {rules_json_path}")
        print(f"Schema path: {schema_path}")
        
        schema_path = self.schema_path
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_all = json.load(f)
        
        no_schema_db=[]
        for item in self.__iter_rules_items(rules_json_path):
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
                no_schema_db.append(db_id)
                continue
            # Build the training prompt
            text = self.__build_io_pair(schema, question, rule_list)
            if max_prompt_length is not None and len(text) > max_prompt_length:
                print(f"Warning: Text length {len(text)} exceeds max_prompt_length {max_prompt_length}")
                continue
            samples.append({"text": text})
        print('no_schema_db:',no_schema_db)
        print(f"Generated {len(samples)} training samples")
        return samples
        
    def tokenize_fn(self,examples, tokenizer):
        outputs = tokenizer(
            examples["text"], 
            padding="max_length", 
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    def load_dataloader(self):
        prompt_records = self.__get_prompts_from_rules(
            tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length,
            rules_json_path=self.rules_json_path,
            schema_path=self.schema_path,
            skip_no_rules=self.skip_no_rules,
            schema_rows=self.schema_rows
        )
        dataset = Dataset.from_list(prompt_records).map(
        lambda x: self.tokenize_fn(x, self.tokenizer),
        batched=True
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        response_template = "generated rules:\n"
        data_collator = CollatorMaskAfterDelimiter(tokenizer=self.tokenizer, delimiter=response_template)


        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        return train_dataloader

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