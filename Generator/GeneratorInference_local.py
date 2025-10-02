# Standard library imports
import argparse
import json
import os
from typing import Dict, List, Optional, Iterable
import time
from datetime import datetime

# PyTorch and deep learning libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# PEFT (Parameter-Efficient Fine-Tuning) libraries
from peft import PeftModel

# Local imports for schema processing
from Process_model.SchemaInformation import SchemaInformation

class Generator:
    def __init__(self, model_name: str, adapter_path: str, bnb_config: Optional[BitsAndBytesConfig], 
                 bf16: bool, tf32: bool, trust_remote_code: bool):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.bnb_config = bnb_config
        self.bf16 = bf16
        self.tf32 = tf32
        self.trust_remote_code = trust_remote_code
    
    def load_tokenizer(self, model_name: str, trust_remote_code: bool) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        """
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    
    def load_model_with_adapter(self, model_name: str, adapter_path: str, bnb_config: Optional[BitsAndBytesConfig], 
                                bf16: bool, tf32: bool, trust_remote_code: bool) -> AutoModelForCausalLM:
        """
        Load the base model and apply LoRA adapter.
        """
    def generate_prompt(self, instruction: str, schema: str, question: str) -> str:
        """Build prompt for SQL generation with rules."""
    prompt = f"""You are a helpful assistant that writes valid SQLite queries.

    You will be given database schema, a question related to the database and some rules.
    You should generate a SQLite query that solves the question with the help of rules.
    The rules contain all the rules you should obey while generating the target SQL, you have to obey all of them.

    Database Schema:
    {schema}

    Question: {question}


    Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.

    SQL:"""
    
    return prompt
