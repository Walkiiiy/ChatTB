"""
Text2SQL Base Experiment Script

This script tests the base text2sql capability of the Qwen model using the BIRD dataset.
It loads questions from train.json, generates SQL using the LLMClient, and verifies
results using SQLTestComparator.

Usage:
    python text2sql_base_experiment.py
"""

import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import our custom classes
from Process_model.LLMClient import LLMClient
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation

base_path = os.path.dirname(os.path.abspath(__file__))

class Text2SQLExperiment:
    """
    Text2SQL base experiment class for testing model capability.
    """
    
    def __init__(self, 
                 train_json_path: str = base_path + "/Spider_dev/res.json",
                 db_root_path: str = base_path + "/Spider_dev/database",
                 output_json_path: str = base_path + "/Spider_dev/EXP_generator_rules_Qwen3_8B.json",
                model='/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B'
                ):
        """
        Initialize the Text2SQL experiment.
        
        Args:
            train_json_path: Path to train.json file
            db_root_path: Path to train_databases directory
            output_json_path: Path to output JSON file
            tables_json_path: Path to train_tables.json file
        """
        self.train_json_path = train_json_path
        self.db_root_path = db_root_path
        self.output_json_path = output_json_path
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.llm_client = LLMClient(model_path=model)
        self.sql_comparator = SQLTestComparator(db_root_path)
        
        # Load data
        self.train_data = self._load_train_data()
        # Load existing results if available
        self.existing_results = self._load_existing_results()
        
        # Statistics - initialize from existing results
        self.total_questions, self.correct_answers, self.failed_executions = self._calculate_existing_statistics()
        
        self.logger.info("Text2SQL experiment initialized successfully")
    
    def _load_train_data(self) -> Dict[str, Any]:
        """Load training data from train.json."""
        try:
            with open(self.train_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} questions from {self.train_json_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load train data: {e}")
            raise
    
    def _load_tables_data(self) -> List[Dict[str, Any]]:
        """Load tables schema data from train_tables.json."""
        try:
            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} database schemas from {self.tables_json_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load tables data: {e}")
            raise
    
    def _load_existing_results(self) -> Dict[str, Any]:
        """Load existing results from output JSON file if it exists."""
        try:
            if os.path.exists(self.output_json_path):
                with open(self.output_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded existing results from {self.output_json_path}")
                return data
            else:
                self.logger.info(f"No existing results file found at {self.output_json_path}")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load existing results: {e}")
            return {}
    
    def _calculate_existing_statistics(self) -> tuple:
        """
        Calculate statistics from existing results.
        
        Returns:
            Tuple of (total_questions, correct_answers, failed_executions)
        """
        total_questions = 0
        correct_answers = 0
        failed_executions = 0
        
        for question_id, result_data in self.existing_results.items():
            if 'output_result' in result_data:
                total_questions += 1
                result = result_data['output_result']
                if result == 1:
                    correct_answers += 1
                elif result == -1:
                    failed_executions += 1
        
        if total_questions > 0:
            accuracy = correct_answers / total_questions
            self.logger.info(f"Found {total_questions} existing results: {correct_answers} correct, {failed_executions} failed (accuracy: {accuracy:.3f})")
        
        return total_questions, correct_answers, failed_executions
    
    def _get_last_processed_index(self) -> int:
        """
        Get the index of the last processed question.
        
        Returns:
            Index of the last processed question, or -1 if no results exist
        """
        if not self.existing_results:
            return -1
        
        # Find the highest numeric key in existing results
        max_index = -1
        for key in self.existing_results.keys():
            try:
                index = int(key)
                # Only count as processed if it has output_result
                if 'output_result' in self.existing_results[key]:
                    max_index = max(max_index, index)
            except ValueError:
                continue
        
        return max_index
    
    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information about resume status.
        
        Returns:                                                        , schema
                                                                ,rules
                                                                )
            Dictionary with resume information
        """
        last_processed = self._get_last_processed_index()
        total_train_questions = len(self.train_data)
        
        info = {
            'has_existing_results': len(self.existing_results) > 0,
            'last_processed_index': last_processed,
            'next_question_index': last_processed + 1 if last_processed >= 0 else 0,
            'total_train_questions': total_train_questions,
            'remaining_questions': max(0, total_train_questions - (last_processed + 1)),
            'existing_statistics': {
                'total_processed': self.total_questions,
                'correct_answers': self.correct_answers,
                'failed_executions': self.failed_executions,
                'accuracy': self.correct_answers / self.total_questions if self.total_questions > 0 else 0
            }
        }
        
        return info
    
        
        if not db_schema:
            self.logger.warning(f"Schema not found for database: {db_id}")
            return ""
        
        # Generate schema description
        schema_parts = []
        table_names = db_schema.get('table_names', [])
        column_names = db_schema.get('column_names_original', [])
        
        # Group columns by table
        table_columns = {}
        for col_info in column_names:
            if col_info[0] == -1:  # Skip wildcard
                continue
            table_idx = col_info[0]
            col_name = col_info[1]
            
            if table_idx < len(table_names):
                table_name = table_names[table_idx]
                if table_name not in table_columns:
                    table_columns[table_name] = []
                table_columns[table_name].append(col_name)
        
        # Generate CREATE TABLE statements
        for table_name, columns in table_columns.items():
            if columns:
                col_defs = ", ".join([f'"{col}"' for col in columns])
                create_stmt = f'CREATE TABLE "{table_name}" ({col_defs})'
                schema_parts.append(create_stmt)
        
        return "\n\n".join(schema_parts)
    
    def _generate_sql_prompt(self, question: str, schema: str, rules: str=None) -> tuple:
        """
        Generate system and user prompts for SQL generation.
        
        Args:
            question: Natural language question
            schema: Database schema information
            rules: Target rules
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """
        You are a helpful assistant that writes valid SQLite queries.
        """
        
        user_prompt = f"""
        you will be given database schema, a question related to the database and some rules.
        you should generate a SQLite query that solve the question with the help of rules.
        the rules contains all the rules you should obey while generating the target sql, you have to obey all of them.
        Database Schema:
        {schema}
        Question: {question}
        {"Rules:"+rules if rules else ""}
        Please generate a SQLite query that answers the question. Return only the SQL query without any explanations or markdown formatting.
        """
        
        # print('\n\n',system_prompt,'\n\n',user_prompt,'\n\n')
        return system_prompt, user_prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL query from model response, handling various formats.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned SQL query
        """
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith('```sql'):
            response = response[6:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        # Clean up whitespace
        response = response.strip()
        
        # Find SQL-like content (starts with SELECT, INSERT, UPDATE, DELETE, etc.)
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                # Stop at semicolon or empty line after SQL
                if line.endswith(';') or (not line and sql_lines):
                    break
        
        if sql_lines:
            sql = ' '.join(sql_lines).strip()
            # Ensure it ends with semicolon
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        return response
    
    def run_experiment(self, max_questions: Optional[int] = None, start_from: Optional[int] = None):
        """
        Run the text2sql experiment.
        
        Args:
            max_questions: Maximum number of questions to process (None for all)
            start_from: Index to start from (None to auto-resume from last result)
        """
        # Auto-resume from last processed result if start_from is not specified
        if start_from is None:
            last_processed = self._get_last_processed_index()
            start_from = last_processed + 1 if last_processed >= 0 else 0
            
            # Show resume information
            resume_info = self.get_resume_info()
            if resume_info['has_existing_results']:
                self.logger.info(f"Resuming experiment from question {start_from}")
                self.logger.info(f"Previous progress: {resume_info['existing_statistics']['total_processed']} questions processed")
                self.logger.info(f"Previous accuracy: {resume_info['existing_statistics']['accuracy']:.3f}")
                self.logger.info(f"Remaining questions: {resume_info['remaining_questions']}")
            else:
                self.logger.info(f"Starting new experiment from question {start_from}")
        else:
            self.logger.info(f"Starting Text2SQL experiment from question {start_from}")
        
        question_ids = list(self.train_data.keys())[start_from:]
        if max_questions:
            question_ids = question_ids[:max_questions]
        
        total_questions = len(question_ids)
        if total_questions == 0:
            self.logger.info("No new questions to process. All questions have been completed.")
            self._print_final_statistics()
            return
        
        self.logger.info(f"Processing {total_questions} new questions (starting from index {start_from})")
        
        # Process each question
        for idx, question_id in enumerate(tqdm(question_ids, desc="Processing questions")):
            try:
                self._process_single_question(question_id, idx + 1, total_questions)
            except Exception as e:
                self.logger.error(f"Error processing question {question_id}: {e}")
                self.failed_executions += 1
                continue
            
            # Save intermediate results every 10 questions
            if (idx + 1) % 10 == 0:
                self._save_results()
        
        # Final save
        self._save_results()
        self._print_final_statistics()
    
    def _process_single_question(self, question_id: str, current_idx: int, total: int):
        """
        Process a single question through the text2sql pipeline.
        
        Args:
            question_id: ID of the question to process
            current_idx: Current question index
            total: Total number of questions
        """
        question_data = self.train_data[question_id]
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        db_id = question_data['db_id']

        rules=question_data['rules']# target rules
        rules=''.join(rules)
        
        print(f"\n{'='*80}")
        print(f"Question {current_idx}/{total} (ID: {question_id})")
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Rules: {rules}")
        # Get database schema
        schema_processor = SchemaInformation()
        schema = schema_processor.generate_schema_info(os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite"))
        if not schema:
            print("❌ No schema found for this database")
            self.failed_executions += 1
            return
        
        # Generate SQL using LLM
        system_prompt, user_prompt = self._generate_sql_prompt(question
                                                                , schema
                                                                ,rules
                                                                )
        
        try:
            response = self.llm_client.chat(user_prompt, system_prompt)
            predicted_sql = self._extract_sql_from_response(response)
            
            print(f"Generated SQL: {predicted_sql}")
            
            # Verify the result
            result = self.sql_comparator.test_sql_with_db_id(
                predicted_sql, ground_truth, db_id
            )
            
            # Update statistics
            self.total_questions += 1
            if result == 1:
                self.correct_answers += 1
                print("✅ Correct!")
            elif result == 0:
                print("❌ Incorrect result")
            else:
                print("❌ Execution error")
                self.failed_executions += 1
            
            # Update the data with new format
            question_data['output_sql'] = predicted_sql
            question_data['output_result'] = result
            
            # Print accuracy so far
            accuracy = self.correct_answers / self.total_questions if self.total_questions > 0 else 0
            print(f"Current Accuracy: {accuracy:.3f} ({self.correct_answers}/{self.total_questions})")
            
        except Exception as e:
            print(f"❌ Error generating SQL: {e}")
            question_data['output_sql'] = f"ERROR: {str(e)}"
            question_data['output_result'] = -1
            self.failed_executions += 1
    
    def _clean_data_for_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the data by removing unnecessary keys and keeping only essential fields.
        
        Args:
            data: Original training data
            
        Returns:
            Cleaned data with only essential fields
        """
        cleaned_data = {}
        
        # Keys to keep in the output
        essential_keys = ['db_id', 'question', 'ground_truth', 'output_sql', 'output_result']
        
        for question_id, question_data in data.items():
            cleaned_question_data = {}
            for key in essential_keys:
                if key in question_data:
                    cleaned_question_data[key] = question_data[key]
            cleaned_data[question_id] = cleaned_question_data
        
        return cleaned_data
    
    def _save_results(self):
        """Save current results to JSON file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_json_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean data before saving
            cleaned_data = self._clean_data_for_output(self.train_data)
            
            # Merge with existing results to preserve all processed data
            merged_data = {**self.existing_results, **cleaned_data}
            
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {self.output_json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _print_final_statistics(self):
        """Print final experiment statistics."""
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Total Questions Processed: {self.total_questions}")
        print(f"Correct Answers: {self.correct_answers}")
        print(f"Failed Executions: {self.failed_executions}")
        
        if self.total_questions > 0:
            accuracy = self.correct_answers / self.total_questions
            print(f"Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"Results saved to: {self.output_json_path}")


def main():
    """Main function to run the experiment."""
    import sys
    
    try:
        # Initialize experiment
        experiment = Text2SQLExperiment()
        
        # Check if user wants to see resume info only
        if len(sys.argv) > 1 and sys.argv[1] == '--resume-info':
            resume_info = experiment.get_resume_info()
            print("\n" + "="*60)
            print("RESUME INFORMATION")
            print("="*60)
            print(f"Has existing results: {resume_info['has_existing_results']}")
            print(f"Last processed index: {resume_info['last_processed_index']}")
            print(f"Next question index: {resume_info['next_question_index']}")
            print(f"Total train questions: {resume_info['total_train_questions']}")
            print(f"Remaining questions: {resume_info['remaining_questions']}")
            
            if resume_info['has_existing_results']:
                stats = resume_info['existing_statistics']
                print(f"\nPrevious Statistics:")
                print(f"  Total processed: {stats['total_processed']}")
                print(f"  Correct answers: {stats['correct_answers']}")
                print(f"  Failed executions: {stats['failed_executions']}")
                print(f"  Accuracy: {stats['accuracy']:.3f} ({stats['accuracy']*100:.1f}%)")
            print("="*60)
            return
        
        # Run experiment (you can modify these parameters)
        experiment.run_experiment(
            # max_questions=50,  # Process first 50 questions for testing
            # start_from=0  # Leave as None to auto-resume from last result
        )
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
