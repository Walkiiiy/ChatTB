# SQL to Natural Language Plan Generator

This module provides a comprehensive solution for converting SQL queries into natural language execution plans using existing LLM models in the project.

## Overview

The `SQLToNLPlanGenerator` class combines existing LLM models (`LLMClient` and `DeepSeekLLMClient`) to generate precise, step-by-step natural language execution plans for SQL queries. This is particularly useful for creating training material for SQL-to-NL plan conversion tasks.

## Features

- **Multiple Model Support**: Works with both local LLM models and DeepSeek API
- **Batch Processing**: Process entire datasets from res.json files
- **Comprehensive SQL Coverage**: Handles complex SQL queries including joins, aggregations, subqueries, window functions, etc.
- **Structured Output**: Generates training material in a consistent JSON format
- **Error Handling**: Robust error handling and logging
- **Progress Tracking**: Real-time progress updates for large datasets

## Installation

The module uses existing dependencies from the project:
- `torch` and `transformers` for local models
- `requests` for DeepSeek API
- Standard Python libraries: `json`, `logging`, `pathlib`, `time`

## Usage

### Basic Usage

```python
from Process_model.SQLToNLPlanGenerator import SQLToNLPlanGenerator

# Initialize with local model
generator = SQLToNLPlanGenerator(
    model_type="local",
    model_path="/path/to/your/model"
)

# Generate NL plan for a single SQL query
sql_query = "SELECT name, age FROM people WHERE age > 30 ORDER BY age DESC LIMIT 5"
nl_plan = generator.generate_nl_plan(sql_query)
print(nl_plan)
```

### Using DeepSeek API

```python
import os

# Initialize with DeepSeek API
generator = SQLToNLPlanGenerator(
    model_type="deepseek",
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
    deepseek_model="deepseek-coder"
)

nl_plan = generator.generate_nl_plan(sql_query)
```

### Processing Datasets

```python
# Process a single dataset
results = generator.process_dataset(
    res_json_path="/path/to/res.json",
    output_path="/path/to/output.json",
    max_queries=100,  # Optional: limit number of queries
    start_index=0     # Optional: start from specific index
)

# Process multiple datasets in batch
dataset_configs = [
    {
        "name": "spider_dev",
        "res_json_path": "/path/to/spider_dev/res.json",
        "max_queries": 50
    },
    {
        "name": "bird_dev",
        "res_json_path": "/path/to/bird_dev/res.json", 
        "max_queries": 30
    }
]

results = generator.process_multiple_datasets(
    dataset_configs=dataset_configs,
    output_dir="/path/to/output/directory"
)
```

## Input Format

The generator expects res.json files with the following structure:

```json
{
  "0": {
    "db_id": "database_name",
    "question": "Natural language question",
    "ground_truth": "SELECT * FROM table WHERE condition",
    "amends": ["Amendment descriptions"],
    "rules": ["Rule descriptions"],
    "amend_res": 1,
    "rule_res": 1,
    "amend_sql": ["SQL queries"],
    "rule_sql": ["SQL queries"]
  }
}
```

## Output Format

The generated output follows this structure:

```json
{
  "metadata": {
    "source_file": "/path/to/input/res.json",
    "total_processed": 100,
    "start_index": 0,
    "max_queries": 100,
    "model_type": "local",
    "timestamp": "2024-01-01 12:00:00"
  },
  "plans": {
    "0": {
      "db_id": "database_name",
      "question": "Natural language question",
      "sql_query": "SELECT * FROM table WHERE condition",
      "nl_plan": "1. Start with all rows from [table].\n2. Keep only rows where [condition].\n3. Return all columns from matching rows.",
      "original_data": { /* original entry data */ }
    }
  }
}
```

## Natural Language Plan Format

The generated plans follow strict formatting rules:

1. **Numbered Steps**: Each step is numbered sequentially
2. **No SQL Keywords**: Uses only plain English descriptions
3. **Square Bracket Notation**: Table and column names in square brackets like `[table.column]`
4. **Precise Descriptions**: Exact logical conditions and operations
5. **Intermediate Sets**: Clearly labeled intermediate result sets
6. **Join Semantics**: Explicit description of join types and matching logic

### Example Output

```
1. Start with all rows from [people].
2. Keep only rows where [age] is greater than 30.
3. For each remaining row, take the values of [name] and [age].
4. Sort those rows by [age] from largest to smallest.
5. Return the first 5 rows.
```

## Configuration Options

### Model Parameters

- `model_type`: "local" or "deepseek"
- `model_path`: Path to local model (required for local type)
- `deepseek_api_key`: API key for DeepSeek (required for deepseek type)
- `deepseek_model`: DeepSeek model name (default: "deepseek-coder")
- `max_new_tokens`: Maximum tokens to generate (default: 1024)
- `temperature`: Sampling temperature (default: 0.1)
- `top_p`: Nucleus sampling parameter (default: 0.9)

### Processing Parameters

- `max_queries`: Maximum number of queries to process (None for all)
- `start_index`: Starting index for processing (default: 0)
- `output_path`: Path to save results
- `output_dir`: Directory for batch processing outputs

## Error Handling

The generator includes comprehensive error handling:

- **Model Loading Errors**: Graceful handling of model initialization failures
- **API Errors**: Retry logic and error reporting for API calls
- **Data Processing Errors**: Individual query failures don't stop batch processing
- **File I/O Errors**: Proper error messages for file access issues

## Logging

The module uses Python's logging system with configurable levels:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Logs include:
- Processing progress updates
- Error messages with context
- Model initialization status
- File I/O operations

## Example Scripts

### Single Query Testing

```python
# Test with various SQL complexities
test_queries = [
    "SELECT name, age FROM people WHERE age > 30",
    "SELECT COUNT(*) FROM users WHERE status = 'active'",
    "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id",
    "SELECT department, AVG(salary) FROM employees GROUP BY department"
]

for sql in test_queries:
    plan = generator.generate_nl_plan(sql)
    print(f"SQL: {sql}")
    print(f"Plan: {plan}\n")
```

### Dataset Processing

```python
# Process Spider dataset
results = generator.process_dataset(
    res_json_path="/home/ubuntu/walkiiiy/ChatTB/Spider_dev/res.json",
    output_path="spider_nl_plans.json",
    max_queries=100
)

# Process Bird dataset  
results = generator.process_dataset(
    res_json_path="/home/ubuntu/walkiiiy/ChatTB/Bird_dev/res.json",
    output_path="bird_nl_plans.json",
    max_queries=50
)
```

## Performance Considerations

- **Local Models**: Faster for batch processing, requires GPU memory
- **API Models**: Slower due to network latency, has rate limits
- **Batch Size**: Process in smaller batches for large datasets
- **Memory Usage**: Monitor GPU memory for local models
- **Rate Limiting**: Add delays between API calls if needed

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model path exists and is accessible
   - Verify sufficient GPU memory
   - Ensure model format is compatible

2. **API Errors**
   - Verify API key is valid and has sufficient credits
   - Check network connectivity
   - Monitor rate limits

3. **Memory Issues**
   - Reduce batch size
   - Use smaller models
   - Process datasets in smaller chunks

4. **Output Format Issues**
   - Check JSON file permissions
   - Verify output directory exists
   - Monitor disk space

## Integration with Existing Project

The `SQLToNLPlanGenerator` integrates seamlessly with existing project components:

- **LLMClient**: Uses existing local model infrastructure
- **DeepSeekLLMClient**: Leverages existing API client
- **Dataset Format**: Compatible with existing res.json structure
- **Logging**: Follows project logging conventions

## Future Enhancements

Potential improvements for future versions:

- Support for additional model types (OpenAI, Anthropic, etc.)
- Parallel processing for faster batch operations
- Custom prompt templates for different SQL dialects
- Integration with database schema information
- Validation of generated plans against original SQL
- Support for more complex SQL features (CTEs, window functions, etc.)

## License

This module is part of the ChatTB project and follows the same licensing terms.
