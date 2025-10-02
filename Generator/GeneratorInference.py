import json
import os
import re
from typing import Dict, Any, Optional, List
from Process_model.DeepSeekLLMClient import DeepSeekLLMClient
from Generator.Tools import Tools


class GeneratorInference:
    def __init__(self, table_schema_path: str, model_path: str, db_root_path: str, adapter_path: str, dataset_path: str):
        """
        Initialize the GeneratorInference class.
        
        Args:
            table_schema_path: Path to table schema information
            model_path: Path to the local LLM model
            db_root_path: Path to database root directory
            adapter_path: Path to model adapter
            dataset_path: Path to condensed_rules dataset
        """
        self.client = DeepSeekLLMClient(api_key=os.environ.get('DEEPSEEK_API_KEY'))
        self.tools = Tools(
            table_schema_path=table_schema_path, 
            model_path=model_path, 
            db_root_path=db_root_path, 
            adapter_path=adapter_path
        )
        self.dataset_path = dataset_path
        self.results = []
    
    def generate_prompt(self, db_id: str, question: str, previous_content: str = None, stage: str = "nlq") -> str:
        """
        Generate appropriate prompt based on the processing stage.
        
        Args:
            db_id: Database identifier
            question: Natural language question
            previous_content: Previous processing results
            stage: Current processing stage (nlq, plan, sql)
            
        Returns:
            Formatted prompt string
        """
        raw_schema = self.tools.get_schema(db_id,1)

        if stage == "nlq":
            return f"""You are a database schema analysis expert. Your task is to analyze the given natural language question as detailed as possible.
You have access to the following tools:
1. get_rules(condition, db_id) - Generate rule operations based on conditions
2. get_schema(db_id, schema_rows) - Get database schema information
3. get_specific_columns_info(db_id, column_names) - Get specific column information
4. get_NL_description(db_id) - Get natural language description of schema

The given question might be ambiguous, contains incorrect schema information, like wrong table names, wrong column names, which may lead to incorrect SQL generation.
your job is to describe the question in the most accurate schema information and as detailed as possible.
If the question is ambiguous, It is recommended to use the get_rules tool to generate rule operations based on conditions.
If the question is clear, all the information is aligned with the schema, you can retuen the question as is.
Use the appropriate tools to gather this information.

Think step by step, and provide a detailed analysis of the question.
Use the tools in thinking phase, and provide the analysis in the end in a <ANALYSIS> tag.
While thinking, when meeting ambiguous information you can't deal with, Format your tool calls as:
<CALL_TOOL>
{{"tool_name": "tool_name", "tool_args": {{"param1": "value1", "param2": "value2"}}}}
</CALL_TOOL>
and yield your analysis in the <ANALYSIS> tag:
<ANALYSIS>
{analysis}
</ANALYSIS>
Now process the following:
Question: {question}
General Schema:
{raw_schema}
"""

        elif stage == "plan":
            return f"""You are a SQL execution planning expert. Based on the natural language question and schema analysis, create a detailed execution plan.

Question: {question}
Database ID: {db_id}

General Schema:
{raw_schema}

Previous Analysis:
{previous_content}

Create a step-by-step natural language execution plan that describes:
1. Which tables to query from
2. How to join tables (if needed)
3. What filtering conditions to apply
4. What aggregations to perform
5. How to order and limit results
6. Expected output format

Be specific about column names, table relationships, and data transformations needed.

Format your response as a clear, numbered list of execution steps."""

        elif stage == "sql":
            return f"""You are a SQL generation expert. Based on the natural language question, schema analysis, and execution plan, generate the corresponding SQL query.

Question: {question}
Database ID: {db_id}

General Schema:
{raw_schema}

Schema Analysis:
{previous_content}

Generate a valid SQL query that:
1. Uses the correct table and column names
2. Implements proper joins where needed
3. Applies appropriate filtering conditions
4. Performs required aggregations
5. Orders and limits results correctly
6. Handles edge cases (null values, data types, etc.)

Return only the SQL query without any additional explanation."""

        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def chat_with_tools(self, user_prompt: str, system_prompt: Optional[str] = None, max_iterations: int = 5) -> str:
        """
        Send prompt to LLM and handle tool calling interactively.
        
        Args:
            user_prompt: User prompt
            system_prompt: Optional system prompt
            max_iterations: Maximum number of tool calling iterations
            
        Returns:
            Final LLM response after all tool calls
        """
        # Clear history and start fresh conversation
        self.client.clear_history()
        
        # Start with initial prompt
        current_prompt = user_prompt
        final_response = ""
        
        iteration = 0
        while iteration < max_iterations:
            # Get response from LLM
            if iteration == 0:
                response = self.client.chat(
                    user_prompt=current_prompt,
                    system_prompt=system_prompt
                )
            else:
                response = self.client.continue_chat(current_prompt)
            
            final_response = response
            
            # Check for tool calls
            tool_calls, errors = self.extract_tool_call(response)
            
            if not tool_calls:
                # No more tool calls, return final response
                break
            
            # Execute tools and prepare next prompt
            tool_results = []
            for tool_call in tool_calls:
                try:
                    tool_name = tool_call.get("tool_name")
                    tool_args = tool_call.get("tool_args", {})
                    result = self.execute_tool(tool_name, tool_args)
                    tool_results.append(f"Tool {tool_name} result: {result}")
                except Exception as e:
                    tool_results.append(f"Error executing {tool_call}: {str(e)}")
            
            # Prepare next prompt with tool results
            if tool_results:
                current_prompt = "Tool execution results:\n" + "\n".join(tool_results) + "\n\nPlease continue your analysis based on these results."
            
            iteration += 1
        
        return final_response
    
    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple chat without tool calling (for backward compatibility).
        
        Args:
            user_prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        response = self.client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return response
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if tool_name == "get_rules":
            return self.tools.get_rules(**tool_args)
        elif tool_name == "get_specific_columns_info":
            return self.tools.get_specific_columns_info(**tool_args)
        elif tool_name == "get_schema":
            return self.tools.get_schema(**tool_args)
        elif tool_name == "get_NL_description":
            return self.tools.get_NL_description(**tool_args)
        else:
            print(f"Invalid tool name: {tool_name}")
            return f"Invalid tool name: {tool_name}, you should consider calling the right tool"
    
    def extract_tool_call(self, text: str, start_token: str = "<CALL_TOOL>", end_token: str = "</CALL_TOOL>") -> tuple:
        """
        Extract tool calls from model response.
        
        Args:
            text: Model response text
            start_token: Start token for tool calls
            end_token: End token for tool calls
            
        Returns:
            Tuple of (tool_calls, errors)
        """
        pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        errors = []

        for match in matches:
            raw_content = match.strip()
            try:
                parsed = json.loads(raw_content)
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                errors.append(raw_content)

        return tool_calls, errors
    
    def NLQ_processing(self, question: str, db_id: str) -> str:
        """
        Process natural language question into schema-level accurate mission description.
        
        Args:
            question: Natural language question
            db_id: Database identifier
            
        Returns:
            Schema analysis and mission description
        """
        print(f"Processing NLQ: {question[:100]}...")
        
        # Generate initial prompt with tool calling instructions
        system_prompt = """You are a database schema analysis expert. You have access to tools to analyze database schemas and understand natural language questions.

Available tools:
1. get_schema(db_id, schema_rows) - Get database schema information
2. get_specific_columns_info(db_id, column_names) - Get specific column information  
3. get_NL_description(db_id) - Get natural language description of schema
4. get_rules(condition, db_id) - Generate rule operations based on conditions

When you need to use a tool, format your call as:
<CALL_TOOL>
{"tool_name": "tool_name", "tool_args": {"param1": "value1", "param2": "value2"}}
</CALL_TOOL>

After receiving tool results, continue your analysis and use additional tools if needed. Provide a comprehensive analysis including:
1. Schema analysis summary
2. Mission description (what exactly needs to be accomplished)
3. Key entities and relationships involved
4. Data requirements and constraints"""

        user_prompt = f"""Analyze the following natural language question and database schema:

Question: {question}
Database ID: {db_id}

Start by getting the schema information you need, then provide a comprehensive analysis."""
        
        # Use interactive tool calling
        response = self.chat_with_tools(user_prompt, system_prompt, max_iterations=5)
        
        return response
    
    def Plan_generate(self, db_id: str, question: str, previous_content: str = None) -> str:
        """
        Generate natural language execution plan.
        
        Args:
            db_id: Database identifier
            question: Natural language question
            previous_content: Previous processing results
            
        Returns:
            Natural language execution plan
        """
        print(f"Generating execution plan for: {question[:100]}...")
        
        system_prompt = """You are a SQL execution planning expert. You have access to tools to analyze database schemas and create detailed execution plans.

Available tools:
1. get_schema(db_id, schema_rows) - Get database schema information
2. get_specific_columns_info(db_id, column_names) - Get specific column information  
3. get_NL_description(db_id) - Get natural language description of schema
4. get_rules(condition, db_id) - Generate rule operations based on conditions

When you need to use a tool, format your call as:
<CALL_TOOL>
{"tool_name": "tool_name", "tool_args": {"param1": "value1", "param2": "value2"}}
</CALL_TOOL>

After receiving tool results, continue your planning and use additional tools if needed. Create a step-by-step natural language execution plan that describes:
1. Which tables to query from
2. How to join tables (if needed)
3. What filtering conditions to apply
4. What aggregations to perform
5. How to order and limit results
6. Expected output format

Be specific about column names, table relationships, and data transformations needed."""

        user_prompt = f"""Create a detailed execution plan for the following question:

Question: {question}
Database ID: {db_id}

Previous Analysis:
{previous_content}

Use tools to get any additional schema information you need, then create a comprehensive execution plan."""
        
        # Use interactive tool calling
        response = self.chat_with_tools(user_prompt, system_prompt, max_iterations=5)
        
        return response
    
    def SQL_generate(self, db_id: str, question: str, previous_content: str = None) -> str:
        """
        Generate SQL query based on plan.
        
        Args:
            db_id: Database identifier
            question: Natural language question
            previous_content: Previous processing results
            
        Returns:
            Generated SQL query
        """
        print(f"Generating SQL for: {question[:100]}...")
        
        system_prompt = """You are a SQL generation expert. You have access to tools to get additional schema information if needed.

Available tools:
1. get_schema(db_id, schema_rows) - Get database schema information
2. get_specific_columns_info(db_id, column_names) - Get specific column information  
3. get_NL_description(db_id) - Get natural language description of schema
4. get_rules(condition, db_id) - Generate rule operations based on conditions

When you need to use a tool, format your call as:
<CALL_TOOL>
{"tool_name": "tool_name", "tool_args": {"param1": "value1", "param2": "value2"}}
</CALL_TOOL>

Generate a valid SQL query that:
1. Uses the correct table and column names
2. Implements proper joins where needed
3. Applies appropriate filtering conditions
4. Performs required aggregations
5. Orders and limits results correctly
6. Handles edge cases (null values, data types, etc.)

Return only the SQL query without any additional explanation."""

        user_prompt = f"""Generate a SQL query for the following:

Question: {question}
Database ID: {db_id}

Previous Analysis and Plan:
{previous_content}

Use tools if you need additional schema information, then generate the SQL query."""
        
        # Use interactive tool calling
        response = self.chat_with_tools(user_prompt, system_prompt, max_iterations=1)
        
        # Clean up the response to extract just the SQL
        sql_query = response.strip()
        
        # Remove any markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
    def Data_iterator(self) -> List[Dict[str, Any]]:
        """
        Load and iterate through the condensed_rules dataset.
        
        Returns:
            List of dataset samples
        """
        print(f"Loading dataset from: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list format for easier iteration
        samples = []
        for key, value in data.items():
            if isinstance(value, dict) and 'db_id' in value and 'question' in value:
                samples.append({
                    'id': key,
                    'db_id': value['db_id'],
                    'question': value['question'],
                    'ground_truth': value.get('ground_truth', ''),
                    'rules': value.get('rules', []),
                    'amends': value.get('amends', [])
                })
        
        print(f"Loaded {len(samples)} samples from dataset")
        return samples
    
    def main_loop(self, max_samples: Optional[int] = None, start_idx: int = 0):
        """
        Main processing loop that iterates through dataset and processes each sample.
        
        Args:
            max_samples: Maximum number of samples to process (None for all)
            start_idx: Starting index for processing
        """
        print("Starting main processing loop...")
        
        # Load dataset
        samples = self.Data_iterator()
        
        # Apply limits
        if max_samples:
            samples = samples[start_idx:start_idx + max_samples]
        else:
            samples = samples[start_idx:]
        
        print(f"Processing {len(samples)} samples starting from index {start_idx}")
        
        for i, sample in enumerate(samples):
            print(f"\n--- Processing Sample {i+1}/{len(samples)} (ID: {sample['id']}) ---")
            
            try:
                # Step 1: NLQ Processing
                print("Step 1: Natural Language Question Processing")
                nlq_result = self.NLQ_processing(sample['question'], sample['db_id'])
                
                # Step 2: Plan Generation
                print("Step 2: Execution Plan Generation")
                plan_result = self.Plan_generate(sample['db_id'], sample['question'], nlq_result)
                
                # Step 3: SQL Generation
                print("Step 3: SQL Query Generation")
                sql_result = self.SQL_generate(sample['db_id'], sample['question'], f"{nlq_result}\n\n{plan_result}")
                
                # Store results
                result = {
                    'id': sample['id'],
                    'db_id': sample['db_id'],
                    'question': sample['question'],
                    'ground_truth': sample['ground_truth'],
                    'nlq_analysis': nlq_result,
                    'execution_plan': plan_result,
                    'generated_sql': sql_result,
                    'rules': sample['rules'],
                    'amends': sample['amends']
                }
                
                self.results.append(result)
                
                print(f"✅ Successfully processed sample {sample['id']}")
                print(f"Generated SQL: {sql_result[:200]}...")
                
            except Exception as e:
                print(f"❌ Error processing sample {sample['id']}: {str(e)}")
                error_result = {
                    'id': sample['id'],
                    'db_id': sample['db_id'],
                    'question': sample['question'],
                    'error': str(e)
                }
                self.results.append(error_result)
        
        print(f"\n--- Processing Complete ---")
        print(f"Successfully processed: {len([r for r in self.results if 'error' not in r])} samples")
        print(f"Errors: {len([r for r in self.results if 'error' in r])} samples")
    
    def save_results(self, output_path: str):
        """
        Save processing results to file.
        
        Args:
            output_path: Path to save results
        """
        print(f"Saving results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved successfully")


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = GeneratorInference(
        table_schema_path="Data/tables.json",  # Adjust path as needed
        model_path="Process_model/models--Qwen3-8B",  # Adjust path as needed
        db_root_path="Data/Databases",  # Adjust path as needed
        adapter_path=None,  # Adjust path as needed
        dataset_path="condensed_rules_all.json"  # Adjust path as needed
    )
    
    # Process a limited number of samples for testing
    generator.main_loop(max_samples=5, start_idx=0)
    
    # Save results
    generator.save_results("generator_results.json")
