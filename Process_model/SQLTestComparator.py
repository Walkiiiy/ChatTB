"""
SQL Test Comparator class for testing if a given SQL query produces the same result as ground truth.
This class provides functionality to execute SQL queries against SQLite databases and compare results.
"""

import sqlite3
import os
import logging
from typing import List, Tuple, Union, Optional, Dict, Any
from pathlib import Path


class SQLTestComparator:
    """
    A class for testing SQL queries against ground truth results.
    
    This class provides methods to:
    - Execute SQL queries against SQLite databases
    - Compare query results with ground truth
    - Handle various data types and encodings
    - Provide detailed error reporting
    """
    
    def __init__(self, db_root_path: str):
        """
        Initialize the SQL Test Comparator.
        
        Args:
            db_root_path: Root path to the directory containing SQLite databases
        """
        self.db_root_path = db_root_path
        self.logger = logging.getLogger(__name__)
        
        # Validate db_root_path exists
        if not os.path.exists(db_root_path):
            raise ValueError(f"Database root path does not exist: {db_root_path}")
    
    def execute_sql(self, predicted_sql: str, ground_truth_sql: str, db_path: str) -> int:
        """
        Execute both predicted SQL and ground truth SQL, then compare results.
        
        Args:
            predicted_sql: The SQL query to test
            ground_truth_sql: The reference SQL query
            db_path: Path to the SQLite database file
            
        Returns:
            1 if results match, 0 if they don't match, -1 if ground truth execution failed
        """
        if not os.path.exists(db_path):
            self.logger.error(f"Database file not found: {db_path}")
            return -1
        
        conn = sqlite3.connect(db_path)
        # Return text as bytes instead of forcing UTF-8 to handle encoding issues
        conn.text_factory = bytes
        cursor = conn.cursor()
        
        try:
            # Execute ground truth SQL first
            cursor.execute(ground_truth_sql)
            ground_truth_res = [
                tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
                for row in cursor.fetchall()
            ]
        except Exception as e:
            self.logger.error(f"Error executing ground truth SQL: {ground_truth_sql}, {e}")
            conn.close()
            return -1
        
        try:
            # Execute predicted SQL
            cursor.execute(predicted_sql)
            predicted_res = [
                tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
                for row in cursor.fetchall()
            ]
        except Exception as e:
            self.logger.error(f"Error executing predicted SQL: {predicted_sql}, {e}")
            conn.close()
            return 0
        
        conn.close()
        
        # Compare results using set operations (order-insensitive)
        result_match = int(set(predicted_res) == set(ground_truth_res))
        
        if result_match:
            self.logger.info("SQL results match!")
        else:
            self.logger.warning("SQL results do not match")
            self.logger.debug(f"Ground truth result count: {len(ground_truth_res)}")
            self.logger.debug(f"Predicted result count: {len(predicted_res)}")
        
        return result_match
    
    def test_sql_with_db_id(self, predicted_sql: str, ground_truth_sql: str, db_id: str) -> int:
        """
        Test SQL query using database ID to construct the database path.
        
        Args:
            predicted_sql: The SQL query to test
            ground_truth_sql: The reference SQL query
            db_id: Database identifier (used to construct db_path)
            
        Returns:
            1 if results match, 0 if they don't match, -1 if ground truth execution failed
        """
        db_path = os.path.join(self.db_root_path, db_id, db_id + '.sqlite')
        return self.execute_sql(predicted_sql, ground_truth_sql, db_path)
    
    def get_database_schema(self, db_path: str, num_sample_rows: Optional[int] = None) -> str:
        """
        Generate a schema description for a database.
        
        Args:
            db_path: Path to the SQLite database
            num_sample_rows: Number of sample rows to include (optional)
            
        Returns:
            String containing the database schema description
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        schema_parts = []
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
                
            # Get table creation SQL
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            create_sql = cursor.fetchone()[0]
            schema_parts.append(create_sql)
            
            # Add sample data if requested
            if num_sample_rows and num_sample_rows > 0:
                cur_table = f"`{table_name}`" if table_name in ['order', 'by', 'group'] else table_name
                cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_sample_rows}")
                column_names = [description[0] for description in cursor.description]
                values = cursor.fetchall()
                
                if values:
                    sample_data = self._format_table_data(column_names, values)
                    schema_parts.append(f"/* \n {num_sample_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_sample_rows}; \n {sample_data} \n */")
        
        conn.close()
        return "\n\n".join(schema_parts)
    
    def _format_table_data(self, column_names: List[str], values: List[Tuple]) -> str:
        """
        Format table data in a readable format.
        
        Args:
            column_names: List of column names
            values: List of row values
            
        Returns:
            Formatted table string
        """
        if not values:
            return ""
        
        # Calculate column widths
        widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
        
        # Create header
        header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
        
        # Create rows
        rows = []
        for value in values:
            row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
            rows.append(row)
        
        return header + '\n' + '\n'.join(rows)
    
    def validate_sql_syntax(self, sql_query: str, db_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax by attempting to prepare the query.
        
        Args:
            sql_query: SQL query to validate
            db_path: Path to the database
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(db_path):
            return False, f"Database file not found: {db_path}"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
            conn.close()
            return True, None
        except sqlite3.Error as e:
            return False, str(e)
    
    def get_query_info(self, sql_query: str, db_path: str) -> Dict[str, Any]:
        """
        Get information about a SQL query execution plan.
        
        Args:
            sql_query: SQL query to analyze
            db_path: Path to the database
            
        Returns:
            Dictionary containing query information
        """
        if not os.path.exists(db_path):
            return {"error": f"Database file not found: {db_path}"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Get query plan
            cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
            plan = cursor.fetchall()
            
            # Get column information
            cursor.execute(sql_query)
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get result count
            cursor.execute(f"SELECT COUNT(*) FROM ({sql_query})")
            count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "query_plan": plan,
                "columns": columns,
                "estimated_rows": count,
                "valid": True
            }
        except sqlite3.Error as e:
            conn.close()
            return {"error": str(e), "valid": False}
    
    def compare_results_detailed(self, predicted_sql: str, ground_truth_sql: str, db_path: str) -> Dict[str, Any]:
        """
        Compare SQL results with detailed information about differences.
        
        Args:
            predicted_sql: The SQL query to test
            ground_truth_sql: The reference SQL query
            db_path: Path to the SQLite database
            
        Returns:
            Dictionary containing detailed comparison results
        """
        if not os.path.exists(db_path):
            return {"error": f"Database file not found: {db_path}"}
        
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        cursor = conn.cursor()
        
        try:
            # Execute ground truth
            cursor.execute(ground_truth_sql)
            ground_truth_res = [
                tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
                for row in cursor.fetchall()
            ]
            ground_truth_set = set(ground_truth_res)
        except Exception as e:
            conn.close()
            return {"error": f"Ground truth execution failed: {e}", "match": False}
        
        try:
            # Execute predicted
            cursor.execute(predicted_sql)
            predicted_res = [
                tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
                for row in cursor.fetchall()
            ]
            predicted_set = set(predicted_res)
        except Exception as e:
            conn.close()
            return {"error": f"Predicted SQL execution failed: {e}", "match": False}
        
        conn.close()
        
        # Detailed comparison
        match = ground_truth_set == predicted_set
        only_in_ground_truth = ground_truth_set - predicted_set
        only_in_predicted = predicted_set - ground_truth_set
        
        return {
            "match": match,
            "ground_truth_count": len(ground_truth_res),
            "predicted_count": len(predicted_res),
            "only_in_ground_truth": list(only_in_ground_truth)[:10],  # Limit to first 10
            "only_in_predicted": list(only_in_predicted)[:10],  # Limit to first 10
            "ground_truth_sample": list(ground_truth_set)[:5],  # Sample results
            "predicted_sample": list(predicted_set)[:5]  # Sample results
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    db_root = "/path/to/your/databases"
    comparator = SQLTestComparator(db_root)
    
    # Test SQL comparison
    predicted_sql = "SELECT * FROM users WHERE age > 25"
    ground_truth_sql = "SELECT * FROM users WHERE age >= 25"
    db_id = "sample_db"
    
    result = comparator.test_sql_with_db_id(predicted_sql, ground_truth_sql, db_id)
    print(f"Comparison result: {result}")
    
    # Get detailed comparison
    db_path = f"{db_root}/{db_id}/{db_id}.sqlite"
    detailed = comparator.compare_results_detailed(predicted_sql, ground_truth_sql, db_path)
    print(f"Detailed comparison: {detailed}")
