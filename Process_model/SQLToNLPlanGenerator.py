"""
SQL to Natural Language Plan Generator

This class combines existing LLM models to generate natural language execution plans
for SQL queries. It processes datasets' res.json files to create training material
for SQL-to-NL plan conversion.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import time

from .LLMClient import LLMClient
from .DeepSeekLLMClient import DeepSeekLLMClient


class SQLToNLPlanGenerator:
    """
    A class that generates natural language execution plans for SQL queries
    using existing LLM models in the project.
    """
    
    def __init__(self, 
                 model_type: str = "local",
                 model_path: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None,
                 deepseek_model: str = "deepseek-coder",
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1,
                 top_p: float = 0.9):
        """
        Initialize the SQL to NL Plan Generator.
        
        Args:
            model_type: Type of model to use ("local" or "deepseek")
            model_path: Path to local model (required for local type)
            deepseek_api_key: DeepSeek API key (required for deepseek type)
            deepseek_model: DeepSeek model name
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize the appropriate model client
        if model_type == "local":
            if not model_path:
                raise ValueError("model_path is required for local model type")
            self.model_client = LLMClient(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        elif model_type == "deepseek":
            if not deepseek_api_key:
                raise ValueError("deepseek_api_key is required for deepseek model type")
            self.model_client = DeepSeekLLMClient(
                api_key=deepseek_api_key,
                model=deepseek_model,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise ValueError("model_type must be 'local' or 'deepseek'")
        
        # System prompt for SQL to NL plan conversion
        self.system_prompt = self._get_system_prompt()
        
        self.logger.info(f"SQLToNLPlanGenerator initialized with {model_type} model")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for SQL to natural language plan conversion."""
        return """You are a precise translator that converts an SQL query into a step-by-step natural-language execution plan.  
Your job: given one SQL query, output a numbered list of steps describing exactly how to execute that query so someone (or another program) can reconstruct the SQL from the description. Accuracy is the top priority. The plan **must not use SQL keywords** (for example: select, from, where, join, group, having, order, limit, distinct, union, intersect, except, exists, in, subquery, alias, as) in the plan itself. Use only ordinary English and precise descriptions.

Strict rules:
1. Output only the numbered plan (one step per line). Do not include commentary, extraneous explanation, or the original SQL. If the SQL is invalid, output a single numbered step that clearly states the error in plain language.
2. Preserve table and column names exactly as they appear in the SQL. When referencing them inside steps, put each name in square brackets like [table] or [table.column] so names remain unambiguous.
3. If the query creates intermediate result sets (derived tables / nested queries), label them explicitly as "intermediate set 1", "intermediate set 2", etc., and describe exactly how each intermediate set is produced and what columns it contains.
4. For filters, use exact logical conditions in plain English, e.g. "keep rows where [age] is greater than 30" or "keep rows where [status] is 'active' and [score] >= 80".
5. For combining rows from multiple tables, describe the combination as: "combine rows from [A] and [B] where [A.col] equals [B.col]" and explicitly state the semantics:
   - If only matching pairs are kept, say: "keep only combinations that have a match in both sides."
   - If all rows from the left side are kept even when there is no match, say: "keep all rows from [left table]; when there is no matching row on the right, leave right-side columns empty."
   - If all rows from the right side are kept, describe symmetrically.
6. For aggregation, avoid the phrase "group by". Instead say: "for each unique value of [column(s)] do the following: compute ..." and list aggregates precisely, e.g. "count of non-empty [id]" or "sum of [amount]" or "average of [duration]". If a filter applies to those aggregated results, say: "after computing those per-value results, keep only those groups where ...".
7. For duplicate elimination, say: "remove duplicate rows so that values of [column list] are unique".
8. For ordering and limiting, say: "sort the final rows by [column] from smallest to largest (or largest to smallest). Then take the first N rows" — be explicit about whether sorting happens before or after limiting.
9. For computed columns or expressions, write the formula in plain math/word form, e.g. "create a new column named 'ratio' equal to [col_a] divided by [col_b]".
10. For boolean logic, express it with plain words ("and", "or", "not") and use parentheses language if needed: "apply both conditions (A and B)".
11. For correlated checks that depend on each row (existence checks), describe them as: "for each row in [X], check whether there exists at least one row in [Y] such that ...; keep those rows where the check succeeds."
12. For set operations (union / intersect / except), describe them in plain terms: "produce rows that are present in either set A or set B (without duplicates)" or "produce rows present in A that are not present in B", etc.
13. For window functions, describe partitioning and ordering in plain English and the computed window metric, e.g. "for each partition defined by [col], compute the running total of [amount] ordered by [date]" and whether the metric is attached to every row or only used for filtering.
14. If query uses naming shortcuts (aliases), show both the original name and the shortcut once, e.g. "[employees] (called E)" and thereafter you may refer to the shortcut in square brackets like [E.id], but still preserve the link to the original.

Output format example (must follow this style exactly):

1. Start with all rows from [people].
2. Keep only rows where [age] is greater than 30.
3. For each remaining row, take the values of [name] and [age].
4. Sort those rows by [age] from largest to smallest.
5. Return the first 5 rows.

Do not use any SQL keywords inside the steps. Be concise but exact. Now convert the SQL query below into the required plan (replace the SQL delimiter with the actual SQL):

-----SQL START-----
<PUT THE SQL QUERY HERE>
-----SQL END-----"""
    
    def generate_nl_plan(self, sql_query: str) -> str:
        """
        Generate a natural language execution plan for a given SQL query.
        
        Args:
            sql_query: The SQL query to convert
            
        Returns:
            Natural language execution plan
        """
        try:
            # Create the user prompt with the SQL query
            user_prompt = f"-----SQL START-----\n{sql_query}\n-----SQL END-----"
            
            # Generate the response using the model client
            response = self.model_client.chat(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating NL plan for SQL: {sql_query[:100]}... Error: {e}")
            return f"1. Error processing SQL query: {str(e)}"
    
    def load_res_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load a res.json file from a dataset.
        
        Args:
            file_path: Path to the res.json file
            
        Returns:
            Dictionary containing the loaded data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} entries from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def process_dataset(self, 
                       res_json_path: str, 
                       output_path: str,
                       max_queries: Optional[int] = None,
                       start_index: int = 0) -> Dict[str, Any]:
        """
        Process a dataset's res.json file to generate natural language plans.
        
        Args:
            res_json_path: Path to the res.json file
            output_path: Path to save the output
            max_queries: Maximum number of queries to process (None for all)
            start_index: Starting index for processing
            
        Returns:
            Dictionary containing processing results
        """
        # Load the dataset
        data = self.load_res_json(res_json_path)
        
        # Filter and limit the data
        entries = list(data.items())[start_index:]
        if max_queries:
            entries = entries[:max_queries]
        
        results = {
            "metadata": {
                "source_file": res_json_path,
                "total_processed": len(entries),
                "start_index": start_index,
                "max_queries": max_queries,
                "model_type": self.model_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "plans": {}
        }
        
        self.logger.info(f"Processing {len(entries)} SQL queries...")
        
        for i, (entry_id, entry_data) in enumerate(entries):
            try:
                # Extract SQL query
                sql_query = entry_data.get('ground_truth', '')
                if not sql_query:
                    self.logger.warning(f"No ground_truth SQL found for entry {entry_id}")
                    continue
                
                # Generate natural language plan
                nl_plan = self.generate_nl_plan(sql_query)
                
                # Store the result
                results["plans"][entry_id] = {
                    "db_id": entry_data.get('db_id', ''),
                    "question": entry_data.get('question', ''),
                    "sql_query": sql_query,
                    "nl_plan": nl_plan,
                    "original_data": entry_data
                }
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(entries)} queries")
                
                # Small delay to avoid overwhelming the model
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error processing entry {entry_id}: {e}")
                results["plans"][entry_id] = {
                    "error": str(e),
                    "original_data": entry_data
                }
        
        # Save results
        self._save_results(results, output_path)
        
        self.logger.info(f"Processing complete. Results saved to {output_path}")
        return results
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save the processing results to a file."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to {output_path}: {e}")
            raise
    
    def process_multiple_datasets(self, 
                                 dataset_configs: List[Dict[str, str]],
                                 output_dir: str) -> Dict[str, Any]:
        """
        Process multiple datasets in batch.
        
        Args:
            dataset_configs: List of dictionaries with 'name', 'res_json_path', 'max_queries' keys
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary containing results from all datasets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {
            "metadata": {
                "total_datasets": len(dataset_configs),
                "model_type": self.model_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "datasets": {}
        }
        
        for config in dataset_configs:
            dataset_name = config['name']
            res_json_path = config['res_json_path']
            max_queries = config.get('max_queries', None)
            start_index = config.get('start_index', 0)
            
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            # Generate output path
            output_path = os.path.join(output_dir, f"{dataset_name}_nl_plans.json")
            
            try:
                # Process the dataset
                results = self.process_dataset(
                    res_json_path=res_json_path,
                    output_path=output_path,
                    max_queries=max_queries,
                    start_index=start_index
                )
                
                all_results["datasets"][dataset_name] = {
                    "output_file": output_path,
                    "metadata": results["metadata"],
                    "total_plans": len(results["plans"])
                }
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
                all_results["datasets"][dataset_name] = {
                    "error": str(e)
                }
        
        # Save summary
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_type": self.model_type,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "model_info": self.model_client.get_model_info()
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Using local model
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Test with a simple SQL query
        test_sql = "SELECT name, age FROM people WHERE age > 30 ORDER BY age DESC LIMIT 5"
        nl_plan = generator.generate_nl_plan(test_sql)
        print("Generated NL Plan:")
        print(nl_plan)
        
    except Exception as e:
        print(f"Local model test failed: {e}")
    
    # Example 2: Using DeepSeek API (if API key is available)
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        try:
            generator = SQLToNLPlanGenerator(
                model_type="deepseek",
                deepseek_api_key=deepseek_api_key
            )
            
            # Test with a simple SQL query
            test_sql = "SELECT COUNT(*) FROM users WHERE status = 'active'"
            nl_plan = generator.generate_nl_plan(test_sql)
            print("Generated NL Plan (DeepSeek):")
            print(nl_plan)
            
        except Exception as e:
            print(f"DeepSeek model test failed: {e}")
    
    # Example 3: Process a dataset
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Process a small subset of Spider dev dataset
        results = generator.process_dataset(
            res_json_path="/home/ubuntu/walkiiiy/ChatTB/Spider_dev/res.json",
            output_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/spider_dev_nl_plans_sample.json",
            max_queries=5  # Process only first 5 queries for testing
        )
        
        print(f"Processed {len(results['plans'])} queries")
        
    except Exception as e:
        print(f"Dataset processing test failed: {e}")
