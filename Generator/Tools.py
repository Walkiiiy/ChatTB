from Process_model.SchemaInformation import SchemaInformation
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from Process_model.DeepSeekLLMClient import DeepSeekLLMClient


logger = logging.getLogger(__name__)

class Tools:
    def __init__(self,table_schema_path: str, model_path: str = "Process_model/models--Qwen3-8B", db_root_path: str = "Data/Databases",adapter_path: str = None):
        """
        Initialize the Tools class with OpenAI client and local LLM model.
        
        Args:
            model_path: Path to the local LLM model
            api_key: OpenAI API key
        """
        self.db_root_path = db_root_path
        self.api_key = os.environ.get('DEEPSEEK_API_KEY')
        self.client = DeepSeekLLMClient(api_key=self.api_key)
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()
        self.schema_info=SchemaInformation(table_schema_path=table_schema_path)
    def _load_model(self):
        """Lazy load the model and tokenizer"""
        try:
            logger.info(f"Loading base model from {self.model_path}")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Load adapter if provided
            if self.adapter_path:
                logger.info(f"Loading adapter from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                self.model = base_model
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            print(f"Failed to load model from {self.model_path}: {e}")
            # Model will be None, handle it in get_rules
    
    def get_rules(self, condition: str, db_id: str) -> str:
        '''
        Using local LLM to generate rule operation based on the rule condition.
        input schema informetion using fetch_linked_schema to generate NL schema information.
        keywords in condition should be quoted.
        Args:
            condition: The rule condition to generate operations for
            db_id: The database id
            
        Returns:
            Generated rule operation as a string
        '''
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Cannot generate rules.")
            raise ValueError("Model not loaded. Please ensure the model path is correct.")
        
        try:
            # Construct the prompt for rule generation
            NL_schema_info,unfound_info=self.schema_info.fetch_linked_schema(db_id,condition)
            if not NL_schema_info:
                logger.error(f"No schema information found for database {db_id}")
                print(f"No schema information found for database {db_id}")
                return unfound_info+"\n\n"+"Cloud not find any schema information"
            prompt = f"""Given the following database schema information:
{NL_schema_info}

And the following rule condition:
{condition}

Generate the appropriate rule operation for this condition."""

            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate the response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            if prompt in generated_text:
                rule_operation = generated_text[len(prompt):].strip()
            else:
                rule_operation = generated_text.strip()
            
            logger.info(f"Successfully generated rule operation for condition: {condition[:50]}...")
            return rule_operation+"\n\n"+unfound_info
            
        except Exception as e:
            logger.error(f"Error generating rules: {e}")
            raise ValueError(f"Failed to generate rules: {e}")

    def get_specific_columns_info(self, db_id: str, column_names: list) -> str:
        '''
        Generate specific column information if database exists
        '''
        logger.info(f"Tool get_specific_column_info: Generating specific column information for {db_id} and {column_names}")
        # Generate specific column information if database exists
        column_info = ""
        if db_id:
            db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                column_info = self.schema_info.generate_specific_column_info(db_path, column_names)
            else:
                logger.error(f"Database {db_id} not found")
                raise ValueError(f"Database {db_id} not found")
        
        return column_info

    def get_schema(self, db_id: str, schema_rows: int=0) -> str:
        '''
        Generate schema information if database exists
        '''
        logger.info(f"Tool get_schema: Generating schema for {db_id}")
        # Generate schema information if database exists
        schema = ""
        if db_id:
            db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    schema = self.schema_info.generate_schema_info(
                        db_path, 
                        num_rows=(schema_rows if schema_rows > 0 else None)
                    )
                except Exception as e:
                    logger.error(f"Warning: Failed to generate schema for {db_id}: {e}")
                    raise ValueError(f"Failed to generate schema: {e}")
        return schema

    def get_NL_description(self, db_id: str) -> str:
        '''
        Using API LLM to generate natural-language description of the schema.
        
        Args:
            schema: The database schema string (CREATE TABLE statements, etc.)
            
        Returns:
            Natural language description of the schema
        '''
        try:
            logger.info("Generating natural language description of schema using OpenAI API")
            
            # Create the prompt for schema description
            system_prompt="You are a helpful database expert. Your task is to analyze database schemas and provide clear, natural language descriptions that explain what the database is about, what tables exist, their purposes, and how they relate to each other."
            user_prompt=f"""Please provide a clear and concise natural language description of the following database schema:

{self.get_schema(db_id)}

Describe:
1. The overall purpose of the database
2. What each table represents
3. Key relationships between tables
4. Important columns and their meanings"""
            
            # Call OpenAI API
            response = self.client.chat(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )            
            # Extract the description
            description = response.strip()
            logger.info("Successfully generated natural language description")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating NL description: {e}")
            raise Exception(f"Failed to generate natural language description: {e}")
