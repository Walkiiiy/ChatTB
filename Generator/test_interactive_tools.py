#!/usr/bin/env python3
"""
Test script for the interactive tool calling functionality in GeneratorInference.py
This demonstrates how tools are called immediately when special tokens are generated.
"""

import os
import sys
from GeneratorInference import GeneratorInference


def test_single_question():
    """Test processing a single question with interactive tool calling."""
    
    # Check if required environment variable is set
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
        print("Please set your DeepSeek API key:")
        print("export DEEPSEEK_API_KEY='your_api_key_here'")
        return
    
    # Initialize the generator
    try:
        generator = GeneratorInference(
            table_schema_path="Bird_dev/dev_tables.json",
            model_path="Process_model/models--Qwen3-8B",
            db_root_path="Bird_dev/database",
            adapter_path=None,
            dataset_path="condensed_rules_all.json"
        )
        print("✅ Generator initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return
    
    # Test question
    question = "What is the highest eligible free rate for K-12 students in Alameda County?"
    db_id = "california_schools"
    
    print(f"\n🔍 Testing Question: {question}")
    print(f"📊 Database: {db_id}")
    
    try:
        # Step 1: NLQ Processing with interactive tool calling
        print("\n--- Step 1: Natural Language Question Processing ---")
        nlq_result = generator.NLQ_processing(question, db_id)
        print(f"✅ NLQ Analysis Complete")
        print(f"📝 Analysis Preview: {nlq_result[:200]}...")
        
        # Step 2: Plan Generation with interactive tool calling
        print("\n--- Step 2: Execution Plan Generation ---")
        plan_result = generator.Plan_generate(db_id, question, nlq_result)
        print(f"✅ Execution Plan Complete")
        print(f"📋 Plan Preview: {plan_result[:200]}...")
        
        # Step 3: SQL Generation with interactive tool calling
        print("\n--- Step 3: SQL Query Generation ---")
        sql_result = generator.SQL_generate(db_id, question, f"{nlq_result}\n\n{plan_result}")
        print(f"✅ SQL Generation Complete")
        print(f"💻 Generated SQL: {sql_result}")
        
        print(f"\n🎉 All steps completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


def test_tool_calling_directly():
    """Test the tool calling mechanism directly."""
    
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
        return
    
    try:
        generator = GeneratorInference(
            table_schema_path="Bird_dev/dev_tables.json",
            model_path="Process_model/models--Qwen3-8B",
            db_root_path="Bird_dev/database",
            adapter_path=None,
            dataset_path="condensed_rules_all.json"
        )
        
        print("\n🔧 Testing Direct Tool Calling")
        
        # Test tool calling with a simple prompt
        system_prompt = """You are a database expert. You have access to tools to get schema information.

Available tools:
1. get_schema(db_id, schema_rows) - Get database schema information
2. get_NL_description(db_id) - Get natural language description of schema

When you need to use a tool, format your call as:
<CALL_TOOL>
{"tool_name": "tool_name", "tool_args": {"param1": "value1", "param2": "value2"}}
</CALL_TOOL>

Get the schema information for the california_schools database."""

        user_prompt = "Please get the schema information for the california_schools database and describe what tables are available."
        
        print("📤 Sending prompt with tool calling instructions...")
        response = generator.chat_with_tools(user_prompt, system_prompt, max_iterations=3)
        
        print(f"📥 Response received:")
        print(f"{response}")
        
    except Exception as e:
        print(f"❌ Error during tool calling test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Testing Interactive Tool Calling in GeneratorInference")
    print("=" * 60)
    
    # Test 1: Direct tool calling
    test_tool_calling_directly()
    
    print("\n" + "=" * 60)
    
    # Test 2: Full pipeline
    test_single_question()
    
    print("\n✅ All tests completed!")
