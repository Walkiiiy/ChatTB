"""
Arctic-Text2SQL-R1 exact-prompt aligned experiment script.

This script follows the exact prompt format from the Arctic-Text2SQL-R1 paper:
- System prompt and User prompt text are filled verbatim (with schema and question).
- Model is asked to output reasoning inside <think>...</think> and final SQL inside
  <answer> ... ```sql ... ``` ... </answer>.
- Only the SQL inside the <answer> block (inside the ```sql``` fence if present)
  is extracted for execution/evaluation.

Usage:
    python arctic_aligned_text2sql_experiment.py
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional
from tqdm import tqdm

# your existing modules (unchanged)
from Process_model.LLMClient import LLMClient
from Process_model.SQLTestComparator import SQLTestComparator
from Process_model.SchemaInformation import SchemaInformation

base_path = os.path.dirname(os.path.abspath(__file__))


class ArcticPromptAlignedExperiment:
    def __init__(
        self,
        input_json_path: str = base_path + "/Spider_dev/res.json",
        db_root_path: str = base_path + "/Spider_dev/database",
        output_json_path: str = base_path + "/Spider_dev/EXP_Arctic-Text2SQL-R1-7B_aligned_rules.json",
        model: str = "/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Arctic-Text2SQL-R1-7B",
        save_every: int = 5,
    ):
        self.input_json_path = input_json_path
        self.db_root_path = db_root_path
        self.output_json_path = output_json_path
        self.save_every = save_every

        # logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # components
        self.llm_client = LLMClient(model_path=model)
        self.sql_comparator = SQLTestComparator(db_root_path)
        self.schema_processor = SchemaInformation()

        # load data
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = json.load(f)

        # results and stats
        self.results: Dict[str, Any] = {}
        self.total = 0
        self.correct = 0
        self.failed_executions = 0

        self.logger.info("Arctic aligned experiment initialized.")

    def _system_prompt(self) -> str:
        # EXACT system prompt from the paper
        return (
            "You are a data science expert. Below, you are provided with a database schema and a natural "
            "language question. Your task is to understand the schema and generate a valid SQL query to "
            "answer the question."
        )

    def _user_prompt(self, schema: str, question_with_evidence: str) -> str:
        # EXACT user prompt structure from the paper (Database Engine: SQLite, Database Schema, Question, Instructions, Output Format)
        user = f"""Database Engine: SQLite

Database Schema:
{schema}

This schema describes the database’s structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question: {question_with_evidence}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
Please provide a detailed chain-of-thought reasoning process and include your thought process within ‘<think>‘ tags. Your final answer should be enclosed within ‘<answer>‘ tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

Correct SQL query here

</answer> """ 
        return user



    def _extract_think_and_sql(self, model_response: str) -> (Optional[str], str):
        """
        Extract <think> content (if any) and SQL from model response.
        Handles both proper <think>/<answer> format and direct SQL generation.
        Returns (think_text or None, extracted_sql_string).
        """
        resp = model_response or ""
        
        # First, try to extract only the assistant's response part
        # Look for the assistant response after the user prompt
        assistant_match = re.search(r"assistant\s*\n(.*?)$", resp, re.S | re.I)
        if assistant_match:
            assistant_response = assistant_match.group(1).strip()
        else:
            # If no "assistant" marker found, use the entire response
            assistant_response = resp
        
        # Try to extract think content (if present)
        think_match = re.search(r"<think>(.*?)</think>", assistant_response, re.S | re.I)
        if think_match:
            think_text = think_match.group(1).strip()
        else:
            # Fallback: extract reasoning content even without proper tags
            think_text = self._extract_reasoning_content(assistant_response)

        # Try to extract answer block first
        answer_match = re.search(r"<answer>(.*?)</answer>", assistant_response, re.S | re.I)
        if answer_match:
            answer_block = answer_match.group(1).strip()
            # Look for SQL in code fences within answer block
            sql_fence_match = re.search(r"```(?:sql)?\s*(.*?)```", answer_block, re.S | re.I)
            if sql_fence_match:
                sql = sql_fence_match.group(1).strip()
            else:
                # Look for SQL statements in answer block
                sql = self._extract_sql_from_text(answer_block)
        else:
            # No answer block found, look for SQL directly in assistant response
            # First try code fences
            sql_fence_match = re.search(r"```(?:sql)?\s*(.*?)```", assistant_response, re.S | re.I)
            if sql_fence_match:
                sql = sql_fence_match.group(1).strip()
            else:
                # Look for SQL statements in the entire assistant response
                sql = self._extract_sql_from_text(assistant_response)

        # Clean up the SQL
        sql = sql.strip()
        if sql and not sql.endswith(";"):
            sql += ";"

        return think_text, sql

    def _extract_sql_from_text(self, text: str) -> str:
        """
        Extract SQL statement from text by looking for SQL keywords.
        Returns the first complete SQL statement found.
        """
        # Look for complete SQL statements
        sql_patterns = [
            # Complete SELECT statements
            r"(SELECT\s+.*?FROM\s+.*?;?)",
            # Complete WITH statements  
            r"(WITH\s+.*?;?)",
            # Complete INSERT/UPDATE/DELETE statements
            r"((?:INSERT|UPDATE|DELETE)\s+.*?;?)",
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, text, re.S | re.I)
            if matches:
                # Return the first complete match
                sql = matches[0].strip()
                # Ensure it ends with semicolon
                if not sql.endswith(";"):
                    sql += ";"
                return sql
        
        # Fallback: look for any line starting with SQL keywords
        lines = text.splitlines()
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", line, re.I):
                in_sql = True
                sql_lines = [line]
            elif in_sql:
                sql_lines.append(line)
                if line.endswith(";"):
                    break
        
        if sql_lines:
            return " ".join(sql_lines)
        
        # Final fallback: return empty string
        return ""

    def _extract_reasoning_content(self, text: str) -> str:
        """
        Extract reasoning/thinking content from model response even when not using <think> tags.
        Looks for explanatory content before the SQL query.
        """
        # Remove the assistant marker if present
        text = re.sub(r"^assistant\s*\n", "", text, flags=re.I)
        
        # Look for content before SQL code blocks or SQL statements
        # Split by common SQL indicators
        sql_indicators = [
            r"```(?:sql)?\s*",  # Code fence
            r"SELECT\s+",       # SELECT statement
            r"WITH\s+",         # WITH statement
            r"INSERT\s+",       # INSERT statement
            r"UPDATE\s+",       # UPDATE statement
            r"DELETE\s+",       # DELETE statement
        ]
        
        reasoning_parts = []
        current_text = text
        
        for pattern in sql_indicators:
            parts = re.split(pattern, current_text, flags=re.I)
            if len(parts) > 1:
                # Take everything before the SQL indicator
                reasoning_part = parts[0].strip()
                if reasoning_part:
                    reasoning_parts.append(reasoning_part)
                break
        
        if reasoning_parts:
            reasoning = reasoning_parts[0]
        else:
            # Fallback: take first few paragraphs or lines before any SQL-like content
            lines = text.split('\n')
            reasoning_lines = []
            in_sql = False
            
            for line in lines:
                line = line.strip()
                if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", line, re.I):
                    in_sql = True
                    break
                if line and not in_sql:
                    reasoning_lines.append(line)
            
            reasoning = '\n'.join(reasoning_lines)
        
        # Clean up the reasoning content
        reasoning = reasoning.strip()
        
        # Remove common prefixes that aren't part of reasoning
        prefixes_to_remove = [
            r"^To translate.*?into.*?query.*?:\s*",
            r"^Let's break.*?down.*?:\s*",
            r"^Here's.*?breakdown.*?:\s*",
        ]
        
        for prefix in prefixes_to_remove:
            reasoning = re.sub(prefix, "", reasoning, flags=re.I)
        
        # Limit length to avoid storing too much
        if len(reasoning) > 2000:
            reasoning = reasoning[:2000] + "..."
        
        return reasoning if reasoning else None

    def _process_one(self, qid: str) -> None:
        qdata = self.data[qid]
        question = qdata.get("question", "").strip()
        evidence = qdata.get("evidence", "") or qdata.get("context", "") or ""
        rules = qdata.get("rules", "")
        # combine evidence + question as paper's {evidence + question}
        if rules:
            question_with_evidence = f"\n{question}\nrules you should follow: {rules}"
        else:
            question_with_evidence = question

        db_id = qdata.get("db_id")
        ground_truth = qdata.get("ground_truth")  # keep same evaluation API as before

        # generate schema
        db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
        schema = self.schema_processor.schema_to_prompt(db_path)
        if not schema:
            self.logger.warning(f"[{qid}] Schema not found for DB {db_id}. Marking as failed execution.")
            qdata["raw_response"] = ""
            qdata["think"] = None
            qdata["output_sql"] = ""
            qdata["output_result"] = -1
            self.failed_executions += 1
            return

        # build prompts exactly as paper
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(schema, question_with_evidence)
        print(user_prompt)
        # call LLM
        response = ""
        try:
            response = self.llm_client.chat(user_prompt, system_prompt)
        except Exception as e:
            self.logger.error(f"[{qid}] LLM chat failed: {e}")
            qdata["raw_response"] = f"ERROR: {e}"
            qdata["think"] = None
            qdata["output_sql"] = ""
            qdata["output_result"] = -1
            self.failed_executions += 1
            return

        # store raw response
        qdata["raw_response"] = response

        # extract think and sql
        think_text, extracted_sql = self._extract_think_and_sql(response)
        qdata["think"] = think_text
        qdata["output_sql"] = extracted_sql

        # evaluate (execution-only)
        try:
            result = self.sql_comparator.test_sql_with_db_id(extracted_sql, ground_truth, db_id)
        except Exception as e:
            self.logger.error(f"[{qid}] Error during evaluation: {e}")
            result = -1

        qdata["output_result"] = result

        # update counters
        self.total += 1
        if result == 1:
            self.correct += 1
        elif result == -1:
            self.failed_executions += 1

        # save to results
        self.results[qid] = qdata

    def run(self, max_questions: Optional[int] = None, start_from: int = 0):
        qids = list(self.data.keys())[start_from:]
        if max_questions:
            qids = qids[:max_questions]

        self.logger.info(f"Starting Arctic-aligned run on {len(qids)} questions (start_from={start_from})")
        print(f"\n🚀 Starting Arctic-aligned experiment on {len(qids)} questions")
        print("=" * 80)

        for idx, qid in enumerate(qids):
            try:
                # Process the question
                self._process_one(qid)
                
                # Calculate current accuracy
                current_acc = self.correct / self.total if self.total > 0 else 0.0
                
                # Get result status
                result = self.data[qid].get("output_result", -1)
                if result == 1:
                    status = "✅ CORRECT"
                elif result == -1:
                    status = "❌ FAILED"
                else:
                    status = "❌ WRONG"
                
                # Show progress with accuracy
                print(f"[{idx+1:3d}/{len(qids)}] QID: {qid} | {status} | "
                      f"Accuracy: {current_acc:.3f} ({current_acc*100:.2f}%) | "
                      f"Correct: {self.correct}/{self.total}")
                
            except Exception as e:
                self.logger.error(f"Unhandled error for {qid}: {e}")
                # try to record failure
                try:
                    self.data[qid]["output_result"] = -1
                    self.results[qid] = self.data[qid]
                except Exception:
                    pass
                self.failed_executions += 1
                
                # Show error in progress
                current_acc = self.correct / self.total if self.total > 0 else 0.0
                print(f"[{idx+1:3d}/{len(qids)}] QID: {qid} | ❌ ERROR | "
                      f"Accuracy: {current_acc:.3f} ({current_acc*100:.2f}%) | "
                      f"Correct: {self.correct}/{self.total}")

            # intermittent save and progress summary
            if (idx + 1) % self.save_every == 0:
                self._save_partial()
                print(f"💾 Saved interim results ({len(self.results)} items)")
                self._print_progress_summary(idx + 1, len(qids))

        # final save + summary
        self._save_partial()
        self._print_summary()

    def _save_partial(self):
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        # keep existing keys + new results: merge so you can resume
        try:
            if os.path.exists(self.output_json_path):
                with open(self.output_json_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
            else:
                prev = {}
        except Exception:
            prev = {}

        merged = {**prev, **self.results}
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved interim results ({len(self.results)} items) to {self.output_json_path}")

    def _print_progress_summary(self, current: int, total: int):
        """Print a progress summary every save_every questions."""
        acc = self.correct / self.total if self.total > 0 else 0.0
        progress_pct = (current / total) * 100
        
        print(f"\n📈 Progress Update: {current}/{total} ({progress_pct:.1f}%)")
        print(f"   Current Accuracy: {acc:.3f} ({acc*100:.2f}%)")
        print(f"   Correct: {self.correct} | Failed: {self.failed_executions} | Wrong: {self.total - self.correct - self.failed_executions}")
        print("-" * 60)

    def _print_summary(self):
        acc = self.correct / self.total if self.total > 0 else 0.0
        print("\n" + "=" * 80)
        print("🎯 ARCTIC-ALIGNED EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"📊 Total evaluated: {self.total}")
        print(f"✅ Correct (execution match): {self.correct}")
        print(f"❌ Failed executions / errors: {self.failed_executions}")
        print(f"🎯 Execution-only Accuracy: {acc:.3f} ({acc*100:.2f}%)")
        print(f"💾 Output saved to: {self.output_json_path}")
        print("=" * 80)
        
        # Add performance indicator
        if acc >= 0.8:
            print("🌟 Excellent performance!")
        elif acc >= 0.6:
            print("👍 Good performance!")
        elif acc >= 0.4:
            print("⚠️  Moderate performance")
        else:
            print("🔧 Needs improvement")
        print("=" * 80)



def main():
    exp = ArcticPromptAlignedExperiment()
    # optional: pass max_questions or start_from
    exp.run()

if __name__ == "__main__":
    main()