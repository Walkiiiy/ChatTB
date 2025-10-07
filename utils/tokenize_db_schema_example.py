"""
Example usage of the database schema tokenizer utility.
"""

from tokenize_db_schema import DatabaseTokenizer
import json

# Example 1: Process Bird_dev dataset
def example_bird_dev():
    tokenizer = DatabaseTokenizer(
        tokenizer_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
    )
    
    tokenizer.process_dataset(
        dataset_root="/home/ubuntu/walkiiiy/ChatTB/Bird_dev/database",
        output_path="/home/ubuntu/walkiiiy/ChatTB/Bird_dev/schema_tokens.json"
    )
    
    # Load and inspect results
    with open("/home/ubuntu/walkiiiy/ChatTB/Bird_dev/schema_tokens.json", 'r') as f:
        results = json.load(f)
    
    print(f"Processed {len(results)} databases")
    for db_name in list(results.keys())[:3]:  # Show first 3
        print(f"\nDatabase: {db_name}")
        print(f"  Tables: {results[db_name]['table_names']}")
        print(f"  Number of table tokens: {len(results[db_name]['table_tokens'])}")
        print(f"  Number of column tokens: {len(results[db_name]['column_tokens'])}")


# Example 2: Process Spider_dev dataset
def example_spider_dev():
    tokenizer = DatabaseTokenizer(
        tokenizer_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
    )
    
    tokenizer.process_dataset(
        dataset_root="/home/ubuntu/walkiiiy/ChatTB/Spider_dev/database",
        output_path="/home/ubuntu/walkiiiy/ChatTB/Spider_dev/schema_tokens.json"
    )


# Example 3: Process a specific database
def example_single_database():
    tokenizer = DatabaseTokenizer(
        tokenizer_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
    )
    
    result = tokenizer.process_database(
        "/home/ubuntu/walkiiiy/ChatTB/Bird_dev/database/california_schools/california_schools.sqlite"
    )
    
    print("California Schools Database:")
    print(f"  Tables: {result['table_names']}")
    print(f"  Columns: {result['column_names'][:10]}...")  # Show first 10
    print(f"  Table tokens: {sorted(list(result['table_tokens']))[:20]}...")  # Show first 20
    print(f"  Column tokens: {sorted(list(result['column_tokens']))[:20]}...")  # Show first 20


if __name__ == "__main__":
    # Run example
    print("=== Example: Processing Bird_dev dataset ===")
    example_bird_dev()







