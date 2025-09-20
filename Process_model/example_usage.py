#!/usr/bin/env python3
"""
Example usage script for SQLToNLPlanGenerator

This script demonstrates how to use the SQLToNLPlanGenerator class to generate
natural language execution plans for SQL queries from various datasets.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))

from Process_model.SQLToNLPlanGenerator import SQLToNLPlanGenerator


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sql_to_nl_plan_generation.log')
        ]
    )


def test_single_query():
    """Test the generator with a single SQL query."""
    print("=" * 60)
    print("Testing Single Query Generation")
    print("=" * 60)
    
    # Initialize generator with local model
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Test queries of varying complexity
        test_queries = [
            "SELECT name, age FROM people WHERE age > 30 ORDER BY age DESC LIMIT 5",
            "SELECT COUNT(*) FROM users WHERE status = 'active'",
            "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id WHERE u.age > 25",
            "SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 50000",
            "SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31' AND customer_id IN (SELECT id FROM customers WHERE city = 'New York')"
        ]
        
        for i, sql in enumerate(test_queries, 1):
            print(f"\nTest Query {i}:")
            print(f"SQL: {sql}")
            print("-" * 40)
            
            nl_plan = generator.generate_nl_plan(sql)
            print(f"Natural Language Plan:\n{nl_plan}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error testing single query: {e}")


def test_deepseek_api():
    """Test the generator with DeepSeek API (if available)."""
    print("\n" + "=" * 60)
    print("Testing DeepSeek API (if available)")
    print("=" * 60)
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("DEEPSEEK_API_KEY not found in environment variables. Skipping DeepSeek test.")
        return
    
    try:
        generator = SQLToNLPlanGenerator(
            model_type="deepseek",
            deepseek_api_key=deepseek_api_key,
            deepseek_model="deepseek-coder"
        )
        
        test_sql = "SELECT name, COUNT(*) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY order_count DESC LIMIT 10"
        
        print(f"SQL: {test_sql}")
        print("-" * 40)
        
        nl_plan = generator.generate_nl_plan(test_sql)
        print(f"Natural Language Plan:\n{nl_plan}")
        
    except Exception as e:
        print(f"Error testing DeepSeek API: {e}")


def process_spider_dev_sample():
    """Process a small sample from Spider dev dataset."""
    print("\n" + "=" * 60)
    print("Processing Spider Dev Dataset Sample")
    print("=" * 60)
    
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Process first 10 queries from Spider dev dataset
        results = generator.process_dataset(
            res_json_path="/home/ubuntu/walkiiiy/ChatTB/Spider_dev/res.json",
            output_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/spider_dev_nl_plans_sample.json",
            max_queries=10,
            start_index=0
        )
        
        print(f"Successfully processed {len(results['plans'])} queries")
        print(f"Results saved to: spider_dev_nl_plans_sample.json")
        
        # Show a sample result
        if results['plans']:
            first_key = list(results['plans'].keys())[0]
            sample = results['plans'][first_key]
            print(f"\nSample Result (Entry {first_key}):")
            print(f"Question: {sample['question']}")
            print(f"SQL: {sample['sql_query']}")
            print(f"NL Plan:\n{sample['nl_plan']}")
        
    except Exception as e:
        print(f"Error processing Spider dev sample: {e}")


def process_bird_dev_sample():
    """Process a small sample from Bird dev dataset."""
    print("\n" + "=" * 60)
    print("Processing Bird Dev Dataset Sample")
    print("=" * 60)
    
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Process first 5 queries from Bird dev dataset
        results = generator.process_dataset(
            res_json_path="/home/ubuntu/walkiiiy/ChatTB/Bird_dev/res.json",
            output_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/bird_dev_nl_plans_sample.json",
            max_queries=5,
            start_index=0
        )
        
        print(f"Successfully processed {len(results['plans'])} queries")
        print(f"Results saved to: bird_dev_nl_plans_sample.json")
        
        # Show a sample result
        if results['plans']:
            first_key = list(results['plans'].keys())[0]
            sample = results['plans'][first_key]
            print(f"\nSample Result (Entry {first_key}):")
            print(f"Question: {sample['question']}")
            print(f"SQL: {sample['sql_query']}")
            print(f"NL Plan:\n{sample['nl_plan']}")
        
    except Exception as e:
        print(f"Error processing Bird dev sample: {e}")


def process_multiple_datasets():
    """Process multiple datasets in batch."""
    print("\n" + "=" * 60)
    print("Processing Multiple Datasets in Batch")
    print("=" * 60)
    
    try:
        generator = SQLToNLPlanGenerator(
            model_type="local",
            model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
        )
        
        # Define dataset configurations
        dataset_configs = [
            {
                "name": "spider_dev_sample",
                "res_json_path": "/home/ubuntu/walkiiiy/ChatTB/Spider_dev/res.json",
                "max_queries": 5,
                "start_index": 0
            },
            {
                "name": "bird_dev_sample", 
                "res_json_path": "/home/ubuntu/walkiiiy/ChatTB/Bird_dev/res.json",
                "max_queries": 3,
                "start_index": 0
            }
        ]
        
        # Process all datasets
        results = generator.process_multiple_datasets(
            dataset_configs=dataset_configs,
            output_dir="/home/ubuntu/walkiiiy/ChatTB/Process_model/batch_output"
        )
        
        print(f"Batch processing complete!")
        print(f"Processed {len(results['datasets'])} datasets")
        
        for dataset_name, dataset_info in results['datasets'].items():
            if 'error' not in dataset_info:
                print(f"- {dataset_name}: {dataset_info['total_plans']} plans generated")
            else:
                print(f"- {dataset_name}: Error - {dataset_info['error']}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")


def main():
    """Main function to run all examples."""
    setup_logging()
    
    print("SQL to Natural Language Plan Generator - Example Usage")
    print("=" * 60)
    
    # Test single query generation
    test_single_query()
    
    # Test DeepSeek API if available
    test_deepseek_api()
    
    # Process sample datasets
    process_spider_dev_sample()
    process_bird_dev_sample()
    
    # Process multiple datasets in batch
    process_multiple_datasets()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the generated JSON files for the natural language plans.")
    print("=" * 60)


if __name__ == "__main__":
    main()
