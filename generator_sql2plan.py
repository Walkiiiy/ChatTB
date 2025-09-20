"""
SQL2Plan Experiment Script

This script converts SQL queries into step-by-step natural-language execution plans.
It loads SQL queries from the dataset, generates execution plans using the LLMClient,
and outputs detailed step-by-step plans that describe how to execute the SQL.

Usage:
    python generator_sql2plan.py
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import our custom classes
from Process_model.LLMClient import LLMClient
from Process_model.SQLTestComparator import SQLTestComparator

base_path = os.path.dirname(os.path.abspath(__file__))

class SQL2PlanExperiment:
    """
    SQL2Plan experiment class for converting SQL queries to execution plans.
    """
    
    def __init__(self, 
                 train_json_path: str = base_path + "/Spider_dev/res.json",
                 db_root_path: str = base_path + "/Spider_dev/database",
                 output_json_path: str = base_path + "/Spider_dev/EXP_sql2plan_Qwen3_8B.json",
                model='/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B'
                ):
        """
        Initialize the SQL2Plan experiment.
        
        Args:
            train_json_path: Path to train.json file
            db_root_path: Path to train_databases directory
            output_json_path: Path to output JSON file
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
        self.total_questions, self.correct_answers, self.failed_executions, self.plan_to_sql_correct = self._calculate_existing_statistics()
        
        self.logger.info("SQL2Plan experiment initialized successfully")
    
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
        plan_to_sql_correct = 0
        
        for question_id, result_data in self.existing_results.items():
            if 'output_result' in result_data:
                total_questions += 1
                result = result_data['output_result']
                if result == 1:
                    correct_answers += 1
                elif result == -1:
                    failed_executions += 1
                
                # Check plan-to-SQL conversion accuracy
                if 'plan_to_sql_result' in result_data and result_data['plan_to_sql_result'] == 1:
                    plan_to_sql_correct += 1
        
        if total_questions > 0:
            accuracy = correct_answers / total_questions
            plan_accuracy = plan_to_sql_correct / total_questions if total_questions > 0 else 0
            self.logger.info(f"Found {total_questions} existing results: {correct_answers} correct, {failed_executions} failed (accuracy: {accuracy:.3f})")
            self.logger.info(f"Plan-to-SQL conversion accuracy: {plan_accuracy:.3f} ({plan_to_sql_correct}/{total_questions})")
        
        return total_questions, correct_answers, failed_executions, plan_to_sql_correct
    
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
                'plan_to_sql_correct': self.plan_to_sql_correct,
                'plan_generation_accuracy': self.correct_answers / self.total_questions if self.total_questions > 0 else 0,
                'plan_to_sql_accuracy': self.plan_to_sql_correct / self.total_questions if self.total_questions > 0 else 0
            }
        }
        
        return info
    
    def _generate_plan_prompt(self, sql_query: str) -> tuple:
        """
        Generate system and user prompts for SQL-to-plan conversion.
        
        Args:
            sql_query: SQL query to convert to execution plan
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """
        You are a precise translator that converts an SQL query into a step-by-step natural-language execution plan.
        """
        
        user_prompt = f"""
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
        {sql_query}
        -----SQL END-----"""
        
        # print('\n\n',system_prompt,'\n\n',user_prompt,'\n\n')
        return system_prompt, user_prompt
    
    def _extract_plan_from_response(self, response: str) -> str:
        """
        Extract execution plan from model response, handling various formats.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned execution plan
        """
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith('```'):
            # Find the end of the code block
            lines = response.split('\n')
            if len(lines) > 1:
                # Remove first line (```) and last line (```) if present
                if lines[-1].strip() == '```':
                    response = '\n'.join(lines[1:-1])
                else:
                    response = '\n'.join(lines[1:])
        
        # Clean up whitespace
        response = response.strip()
        
        # Find numbered plan content (starts with numbers like '1.', '2.', etc.)
        lines = response.split('\n')
        plan_lines = []
        in_plan = False
        
        for line in lines:
            line = line.strip()
            # Check if line starts with a number followed by a dot
            if line and line[0].isdigit() and '.' in line[:3]:
                in_plan = True
            if in_plan:
                plan_lines.append(line)
                # Stop at empty line after plan or non-numbered content
                if not line and plan_lines:
                    break
        
        if plan_lines:
            return '\n'.join(plan_lines).strip()
        
        return response
    
    def _generate_plan_to_sql_prompt(self, execution_plan: str) -> tuple:
        """
        Generate system and user prompts for converting execution plan back to SQL.
        
        Args:
            execution_plan: Natural language execution plan
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """
        You are a precise translator that converts a natural-language execution plan back into a valid SQL query.
        """
        
        user_prompt = f"""
        Your job: given a step-by-step natural-language execution plan, construct the corresponding SQL query.
        The plan describes exactly how to execute a query, and you need to convert it to SQL.
        
        Guidelines:
        1. Use standard SQL syntax (SQLite compatible)
        2. Preserve table and column names exactly as they appear in the plan (without square brackets)
        3. Convert natural language operations back to SQL keywords
        4. Maintain the logical structure and order of operations
        5. Return only the SQL query without explanations or markdown formatting
        
        Execution Plan:
        {execution_plan}
        
        Please convert this execution plan back to a SQL query:"""
        
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
                self.logger.info(f"Previous plan generation success rate: {resume_info['existing_statistics']['plan_generation_accuracy']:.3f}")
                self.logger.info(f"Previous plan-to-SQL conversion accuracy: {resume_info['existing_statistics']['plan_to_sql_accuracy']:.3f}")
                self.logger.info(f"Remaining questions: {resume_info['remaining_questions']}")
            else:
                self.logger.info(f"Starting new experiment from question {start_from}")
        else:
            self.logger.info(f"Starting SQL2Plan experiment from question {start_from}")
        
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
        Process a single question through the SQL-to-plan pipeline.
        
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
        print(f"Ground Truth SQL: {ground_truth}")
        print(f"Rules: {rules}")
        
        # Generate execution plan using LLM
        system_prompt, user_prompt = self._generate_plan_prompt(ground_truth)
        
        try:
            # Step 1: Generate execution plan from SQL
            response = self.llm_client.chat(user_prompt, system_prompt)
            execution_plan = self._extract_plan_from_response(response)
            
            print(f"Generated Execution Plan:")
            print(execution_plan)
            
            # Step 2: Convert execution plan back to SQL
            plan_to_sql_system, plan_to_sql_user = self._generate_plan_to_sql_prompt(execution_plan)
            sql_response = self.llm_client.chat(plan_to_sql_user, plan_to_sql_system)
            converted_sql = self._extract_sql_from_response(sql_response)
            
            print(f"\nConverted SQL from Plan:")
            print(converted_sql)
            
            # Step 3: Verify the converted SQL against ground truth
            verification_result = self.sql_comparator.test_sql_with_db_id(
                converted_sql, ground_truth, db_id
            )
            
            # Update statistics
            self.total_questions += 1
            self.correct_answers += 1  # Plan generation success
            
            if verification_result == 1:
                self.plan_to_sql_correct += 1
                print("✅ Plan-to-SQL conversion successful!")
            elif verification_result == 0:
                print("❌ Plan-to-SQL conversion: Incorrect result")
            else:
                print("❌ Plan-to-SQL conversion: Execution error")
            
            # Update the data with new format
            question_data['output_plan'] = execution_plan
            question_data['output_sql_from_plan'] = converted_sql
            question_data['output_result'] = 1  # Plan generation success
            question_data['plan_to_sql_result'] = verification_result
            
            # Print progress so far
            plan_accuracy = self.correct_answers / self.total_questions if self.total_questions > 0 else 0
            conversion_accuracy = self.plan_to_sql_correct / self.total_questions if self.total_questions > 0 else 0
            print(f"Current Plan Generation Success Rate: {plan_accuracy:.3f} ({self.correct_answers}/{self.total_questions})")
            print(f"Current Plan-to-SQL Conversion Accuracy: {conversion_accuracy:.3f} ({self.plan_to_sql_correct}/{self.total_questions})")
            
        except Exception as e:
            print(f"❌ Error in plan generation or conversion: {e}")
            question_data['output_plan'] = f"ERROR: {str(e)}"
            question_data['output_sql_from_plan'] = f"ERROR: {str(e)}"
            question_data['output_result'] = -1
            question_data['plan_to_sql_result'] = -1
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
        essential_keys = ['db_id', 'question', 'ground_truth', 'output_plan', 'output_sql_from_plan', 'output_result', 'plan_to_sql_result']
        
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
        print("SQL2PLAN EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Total Questions Processed: {self.total_questions}")
        print(f"Successful Plan Generations: {self.correct_answers}")
        print(f"Successful Plan-to-SQL Conversions: {self.plan_to_sql_correct}")
        print(f"Failed Executions: {self.failed_executions}")
        
        if self.total_questions > 0:
            plan_success_rate = self.correct_answers / self.total_questions
            conversion_accuracy = self.plan_to_sql_correct / self.total_questions
            print(f"Plan Generation Success Rate: {plan_success_rate:.3f} ({plan_success_rate*100:.1f}%)")
            print(f"Plan-to-SQL Conversion Accuracy: {conversion_accuracy:.3f} ({conversion_accuracy*100:.1f}%)")
        
        print(f"Results saved to: {self.output_json_path}")


def main():
    """Main function to run the experiment."""
    import sys
    
    try:
        # Initialize experiment
        experiment = SQL2PlanExperiment(
            train_json_path=base_path + "/Spider_dev/res.json",
            db_root_path= base_path + "/Spider_dev/database",
            output_json_path= base_path + "/Spider_dev/sql2plan_Qwen3_8B.json",
            model='/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B'
        )
        
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
                print(f"  Successful plan generations: {stats['correct_answers']}")
                print(f"  Successful plan-to-SQL conversions: {stats['plan_to_sql_correct']}")
                print(f"  Failed executions: {stats['failed_executions']}")
                print(f"  Plan generation success rate: {stats['plan_generation_accuracy']:.3f} ({stats['plan_generation_accuracy']*100:.1f}%)")
                print(f"  Plan-to-SQL conversion accuracy: {stats['plan_to_sql_accuracy']:.3f} ({stats['plan_to_sql_accuracy']*100:.1f}%)")
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
