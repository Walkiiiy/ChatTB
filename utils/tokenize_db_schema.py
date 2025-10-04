"""
Utility script to tokenize database table and column names using Qwen3-8B tokenizer.
Processes all databases in a given root directory and saves tokenized schema information.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Set, List
from pathlib import Path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseTokenizer:
    """Tokenizes database schemas using a specified tokenizer."""
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize the DatabaseTokenizer with a specific tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer model (e.g., models--Qwen3-8B)
        """
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        logger.info("Tokenizer loaded successfully")
    
    def extract_schema_from_db(self, db_path: str) -> Dict[str, List[str]]:
        """
        Extract table names and column names from a SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
            
        Returns:
            Dictionary with 'tables' and 'columns' lists
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
            
            # Get all column names from all tables
            columns = []
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                table_columns = [col[1] for col in cursor.fetchall()]
                columns.extend(table_columns)
            
            conn.close()
            
            return {
                'tables': tables,
                'columns': columns
            }
        except Exception as e:
            logger.error(f"Error extracting schema from {db_path}: {e}")
            return {'tables': [], 'columns': []}
    
    def tokenize_names(self, names: List[str]) -> Set[int]:
        """
        Tokenize a list of names and return unique token IDs.
        
        Args:
            names: List of table or column names
            
        Returns:
            Set of unique token IDs
        """
        all_tokens = set()
        
        for name in names:
            # Tokenize the name
            tokens = self.tokenizer.encode(name, add_special_tokens=False)
            all_tokens.update(tokens)
        
        return all_tokens
    
    def process_database(self, db_path: str) -> Dict[str, Set[int]]:
        """
        Process a single database and return tokenized schema.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Dictionary with 'table_tokens' and 'column_tokens' sets
        """
        logger.info(f"Processing database: {db_path}")
        
        schema = self.extract_schema_from_db(db_path)
        
        table_tokens = self.tokenize_names(schema['tables'])
        column_tokens = self.tokenize_names(schema['columns'])
        
        logger.info(f"  Found {len(schema['tables'])} tables, {len(schema['columns'])} columns")
        logger.info(f"  Generated {len(table_tokens)} unique table tokens, {len(column_tokens)} unique column tokens")
        
        return {
            'table_tokens': table_tokens,
            'column_tokens': column_tokens,
            'table_names': schema['tables'],
            'column_names': schema['columns']
        }
    
    def process_dataset(self, dataset_root: str, output_path: str) -> None:
        """
        Process all databases in a dataset directory and save tokenized results.
        
        Args:
            dataset_root: Root path to database directory (e.g., "Bird_dev/database")
            output_path: Path to save the output JSON file
        """
        dataset_path = Path(dataset_root)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset root path does not exist: {dataset_root}")
        
        logger.info(f"Processing dataset at: {dataset_root}")
        
        results = {}
        
        # Find all SQLite database files
        db_files = list(dataset_path.glob("**/*.sqlite"))
        
        if not db_files:
            logger.warning(f"No SQLite databases found in {dataset_root}")
            return
        
        logger.info(f"Found {len(db_files)} database(s)")
        
        for db_file in db_files:
            # Extract database name (parent directory name)
            db_name = db_file.parent.name
            
            # Process the database
            tokenized_data = self.process_database(str(db_file))
            
            # Convert sets to lists for JSON serialization
            results[db_name] = {
                'tokens': sorted(list(tokenized_data['table_tokens'])+list(tokenized_data['column_tokens'])),
                'table_names': tokenized_data['table_names'],
                'column_names': list(set(tokenized_data['column_names']))  # Remove duplicates
            }
        
        # Save results to JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total databases processed: {len(results)}")


def main():
    """Main function to run the tokenizer utility."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tokenize database table and column names using Qwen3-8B tokenizer"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B",
        help="Path to the tokenizer model (default: models--Qwen3-8B)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root path to database directory (e.g., Bird_dev/database)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = DatabaseTokenizer(args.tokenizer_path)
    
    # Process dataset
    tokenizer.process_dataset(args.dataset_root, args.output)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()

