#!/usr/bin/env python3
"""
SQL2Plan Experiment Script - Second Refinement

This updated script addresses issues observed with small or brittle LLMs that
returned malformed SQL (e.g. "...") and logged warnings that generation flags
were ignored. Changes made in this version:

 - Removed passing unsupported generation flags (temperature/top_k/top_p) to the LLM client.
 - Strengthened few-shot examples for both SQL->Plan and Plan->SQL directions.
 - Added strict validation for both plan output and SQL output (rejects `...`, ensures SQL contains key tokens).
 - Added re-prompt loops with corrective messages when the model produces invalid outputs.
 - Improved logging and clearer error messages.

Usage:
    python sql2plan_refined_v2.py [--max N] [--start S] [--resume-info]

Note: keep your LLMClient and SQLTestComparator available in Process_model.
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from Process_model.LLMClient import LLMClient
from Process_model.SQLTestComparator import SQLTestComparator

base_path = os.path.dirname(os.path.abspath(__file__))

# ------- Configuration -------
DEFAULT_MODEL_PATH = '/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B'
PLAN_JSON_KEY = 'execution_plan'
LLM_RETRIES = 3
PLAN_GENERATION_RETRY_PROMPT = (
    "The previous response did not follow the required JSON format. "
    "Return ONLY a JSON object with a single key \"execution_plan\" whose value is an array of textual steps. "
    "Each step must avoid SQL keywords and must reference table/column names using double square brackets, e.g. [[table]] and [[column]]."
)

# ------- Utilities -------

def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _find_first_json_with_key(text: str, key: str) -> Optional[Tuple[int, int, dict]]:
    # Try code-fenced JSON first
    code_fence_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
    for m in code_fence_pattern.finditer(text):
        candidate = m.group(1).strip()
        obj = _safe_json_loads(candidate)
        if isinstance(obj, dict) and key in obj:
            return (m.start(1), m.end(1), obj)

    # Try any JSON-like substring
    greedy_pattern = re.compile(r'(\{(?:.|\n)*?\})', re.DOTALL)
    for m in greedy_pattern.finditer(text):
        candidate = m.group(1)
        obj = _safe_json_loads(candidate)
        if isinstance(obj, dict) and key in obj:
            return (m.start(1), m.end(1), obj)

    return None


def _looks_like_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    s = sql.strip()
    # Reject placeholder-only outputs
    if re.fullmatch(r'[.\s]+', s):
        return False
    # Must contain at least one SQL keyword or common patterns (SELECT, WITH, INSERT, UPDATE, DELETE, COUNT, FROM)
    if re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE|COUNT|FROM)\b', s, flags=re.IGNORECASE):
        # also avoid '...'
        if '...' in s:
            return False
        return True
    return False

# ------- Experiment class (v2) -------
class SQL2PlanExperiment:
    def __init__(
        self,
        train_json_path: str = base_path + "/Spider_dev/res.json",
        db_root_path: str = base_path + "/Spider_dev/database",
        output_json_path: str = base_path + "/Spider_dev/EXP_sql2plan_Qwen3_8B_refined_v2.json",
        model: str = DEFAULT_MODEL_PATH,
        retries: int = LLM_RETRIES,
    ):
        self.train_json_path = train_json_path
        self.db_root_path = db_root_path
        self.output_json_path = output_json_path
        self.model_path = model
        self.retries = retries

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.llm_client = LLMClient(model_path=self.model_path)
        self.sql_comparator = SQLTestComparator(db_root_path)

        self.train_data = self._load_train_data()
        if isinstance(self.train_data, list):
            self.train_data = {str(i): v for i, v in enumerate(self.train_data)}

        self.existing_results = self._load_existing_results()
        self.total_questions, self.correct_answers, self.failed_executions, self.plan_to_sql_correct = self._calculate_existing_statistics()

        self.logger.info("SQL2PlanExperiment v2 initialized")

    def _load_train_data(self) -> Dict[str, Any]:
        try:
            with open(self.train_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} training examples from {self.train_json_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load train data: {e}")
            raise

    def _load_existing_results(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.output_json_path):
                with open(self.output_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded existing results from {self.output_json_path}")
                return data
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load existing results: {e}")
            return {}

    def _calculate_existing_statistics(self) -> tuple:
        total_questions = 0
        correct_answers = 0
        failed_executions = 0
        plan_to_sql_correct = 0
        for qid, result in self.existing_results.items():
            if isinstance(result, dict) and 'output_result' in result:
                total_questions += 1
                r = result.get('output_result')
                if r == 1:
                    correct_answers += 1
                elif r == -1:
                    failed_executions += 1
                if result.get('plan_to_sql_result') == 1:
                    plan_to_sql_correct += 1
        if total_questions > 0:
            self.logger.info(f"Found {total_questions} existing results: {correct_answers} correct, {failed_executions} failed")
        return total_questions, correct_answers, failed_executions, plan_to_sql_correct

    def _get_last_processed_index(self) -> int:
        if not self.existing_results:
            return -1
        max_index = -1
        for key, val in self.existing_results.items():
            try:
                idx = int(key)
                if isinstance(val, dict) and 'output_result' in val:
                    max_index = max(max_index, idx)
            except Exception:
                continue
        return max_index

    def get_resume_info(self) -> Dict[str, Any]:
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
                'plan_to_sql_accuracy': self.plan_to_sql_correct / self.total_questions if self.total_questions > 0 else 0,
            }
        }
        return info

    # -------- LLM wrapper (avoid unsupported kwargs) --------
    def _call_llm(self, user_prompt: str, system_prompt: str) -> str:
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                # Call the LLM client with only positional args to avoid unsupported flags.
                resp = self.llm_client.chat(user_prompt, system_prompt)
                if not isinstance(resp, str):
                    if isinstance(resp, dict) and 'text' in resp:
                        return resp['text']
                    resp = str(resp)
                return resp
            except Exception as e:
                last_exc = e
                self.logger.warning(f"LLM call failed on attempt {attempt}/{self.retries}: {e}")
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"LLM calls failed after {self.retries} attempts: {last_exc}")

    # -------- Prompt generation with few-shot examples --------
    def _generate_plan_prompt(self, sql_query: str) -> Tuple[str, str]:

        system_prompt = (
            "You are a precise translator that converts an SQL query into a step-by-step natural-language execution plan. "
            "Return ONLY a single JSON object with key \"execution_plan\" mapped to an array of steps. Do NOT return any extra text.\n\n"
            "GUIDELINES (must follow exactly):\n"
            "1. Output exactly one JSON object and nothing else (no markdown, no commentary, no code fences).\n"
            "2. The JSON must contain a single key (use the PLAN_JSON_KEY constant in examples) whose value is an array of ordered steps.\n"
            "3. Each step must be an atomic, deterministic action that together allow reconstruction of the SQL (examples: scan table, apply predicate, compute aggregate, compute join match, sort, take top N, remove duplicates).\n"
            "4. Do NOT use SQL reserved words inside steps: select, from, where, join, group, having, order, limit, distinct, union, intersect, except. Use alternative verbs such as scan, filter, combine, aggregate, sort, take top, remove duplicates.\n"
            "5. Mark table/alias/column identifiers exactly as they appear in the SQL using double-square-brackets: [[table]], [[table.column]], [[alias.column]], or [[\"weird column name\"]].\n"
            "6. If the SQL uses aliases, declare them early in a step using this exact pattern: 'Treat [[alias=original_table]] as shorthand.' Afterwards reference alias-qualified columns like [[s.name]].\n"
            "7. When you need to refer to intermediate/derived relations, name them sequentially as T1, T2, T3 and reference those names in later steps.\n"
            "8. For expressions, functions and aggregates include the exact expression inside double-square-brackets, e.g. [[COUNT(*)]] or [[salary * 1.1]].\n"
            "9. For subqueries or CTEs produce their plan steps first and label them as 'Subplan S1:' (then subsequent lines for S1 if needed). Reference the derived subplan by [[S1]] in later steps.\n"
            "10. Preserve boolean logic and parentheses exactly in predicates (for example: '(A AND B) OR (C AND NOT D)').\n"
            "11. Keep steps concise (prefer <= 140 characters) and unambiguous about which rows are kept or dropped.\n"
            "12. Final steps must explicitly state which columns or computed values are produced as the query output.\n"
            "13. If parsing fails, return a single-step array whose only element starts with 'ERROR:' followed by a short parse message.\n"
        )

        # Few-shot examples (show exact expected JSON output format)
        ex_sql_1 = "SELECT name FROM students WHERE age > 18;"
        ex_plan_1 = {PLAN_JSON_KEY: [
            "Scan rows from [[students]] and keep those where [[students.age]] > 18.",
            "For each remaining row output [[students.name]]."
        ]}

        ex_sql_2 = "SELECT COUNT(*) FROM singer;"
        ex_plan_2 = {PLAN_JSON_KEY: [
            "Access all rows of [[singer]].",
            "Compute [[COUNT(*)]] over the accessed rows and return the single aggregate result."
        ]}

        ex_sql_3 = (
            "SELECT s.name, COUNT(*) AS songs "
            "FROM singer s JOIN album a ON s.id = a.singer_id "
            "WHERE a.year >= 2000 "
            "GROUP BY s.name HAVING COUNT(*) > 5 ORDER BY songs DESC LIMIT 10;"
        )
        ex_plan_3 = {PLAN_JSON_KEY: [
            "Treat [[s=singer]] and [[a=album]] as shorthand.",
            "Combine rows from [[s]] and [[a]] where [[s.id]] = [[a.singer_id]]; keep only rows that have matches in both sides.",
            "Keep only rows where [[a.year]] >= 2000.",
            "For each unique value of [[s.name]] compute [[COUNT(*)]] and label it [[songs]].",
            "Keep only groups where [[songs]] > 5.",
            "Sort the groups by [[songs]] descending and keep only the first 10 groups.",
            "Output [[s.name]] and [[songs]] for each remaining group."
        ]}

        ex_sql_4 = (
            "WITH recent_albums AS (SELECT * FROM album WHERE release_date > '2020-01-01') "
            "SELECT r.title FROM recent_albums r WHERE r.rating >= 4.5;"
        )
        ex_plan_4 = {PLAN_JSON_KEY: [
            "Subplan S1: Scan rows from [[album]] and keep those where [[album.release_date]] > '2020-01-01'; label this derived relation [[S1]] (recent_albums).",
            "Treat [[r=S1]] as shorthand for the derived relation.",
            "Keep rows from [[r]] where [[r.rating]] >= 4.5.",
            "Output [[r.title]] from the remaining rows."
        ]}

        user_prompt = (
            "Given a single SQL query, return ONLY a JSON object with one key \"execution_plan\" whose value is an array of short, ordered steps.\n\n"
            "Each step must:\n"
            "- Avoid SQL reserved keywords and be plain, deterministic, and actionable.\n"
            "- Use double-square-bracket annotations for tables, aliases, columns and exact expressions.\n"
            "- Be atomic so that the sequence of steps can be used to reconstruct the original SQL.\n\n"
        )

        user_prompt += f"Example SQL:\n{ex_sql_1}\nDesired JSON:\n{json.dumps(ex_plan_1, ensure_ascii=False, indent=2)}\n\n"
        user_prompt += f"Example SQL:\n{ex_sql_2}\nDesired JSON:\n{json.dumps(ex_plan_2, ensure_ascii=False, indent=2)}\n\n"
        user_prompt += f"Example SQL (join/aggregate/order/limit):\n{ex_sql_3}\nDesired JSON:\n{json.dumps(ex_plan_3, ensure_ascii=False, indent=2)}\n\n"
        user_prompt += f"Example SQL (CTE / subplan):\n{ex_sql_4}\nDesired JSON:\n{json.dumps(ex_plan_4, ensure_ascii=False, indent=2)}\n\n"
        user_prompt += "Now convert the following SQL into the JSON plan (no extra text):\n"
        user_prompt += f"{sql_query}\n"

        return system_prompt, user_prompt

    def _extract_plan_from_response(self, response: str) -> List[str]:
        if not response or not isinstance(response, str):
            raise ValueError("Empty model response")

        found = _find_first_json_with_key(response, PLAN_JSON_KEY)
        if found:
            _, _, obj = found
            steps = obj.get(PLAN_JSON_KEY)
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return [s.strip() for s in steps]
            else:
                raise ValueError("Found JSON but 'execution_plan' is not a list of strings")

        # Fallback: parse numbered list after 'Execution Plan:'
        fence_match = re.search(r'Execution Plan:\s*(?:```(?:.*?)```\s*)?(.*)$', response, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            tail = fence_match.group(1).strip()
            lines = re.split(r'\n+', tail)
            steps = []
            numbered = False
            for line in lines:
                m = re.match(r'\s*\d+\.|^\s*-\s+', line)
                if m:
                    numbered = True
                    step = re.sub(r'^\s*\d+\.?\s*', '', line).strip()
                    step = re.sub(r'^\s*-\s*', '', step).strip()
                    if step:
                        steps.append(step)
            if numbered and steps:
                return steps

        # Last resort: try to parse any top-level JSON-like area
        json_like = re.search(r'\{(?:.|\n)*\}', response, flags=re.DOTALL)
        if json_like:
            obj = _safe_json_loads(json_like.group(0))
            if isinstance(obj, dict) and PLAN_JSON_KEY in obj and isinstance(obj[PLAN_JSON_KEY], list):
                return [str(s).strip() for s in obj[PLAN_JSON_KEY]]

        raise ValueError("Could not extract an execution plan in the required JSON format from model response")

    # -------- Plan generation with validation and retries --------
    def _plan_generation_with_validation(self, sql_query: str) -> List[str]:
        system_prompt, user_prompt = self._generate_plan_prompt(sql_query)

        for attempt in range(1, self.retries + 1):
            response = self._call_llm(user_prompt, system_prompt)
            try:
                steps = self._extract_plan_from_response(response)
                if not steps or not isinstance(steps, list):
                    raise ValueError("Plan is empty or not a list")
                if not any(re.search(r'\[\[[^\]]+\]\]', s) for s in steps):
                    raise ValueError("Plan does not contain any double-square-bracketed identifiers like [[table]] or [[column]]")
                return steps
            except Exception as e:
                self.logger.warning(f"Plan extraction/validation failed on attempt {attempt}: {e}")
                if attempt < self.retries:
                    user_prompt = PLAN_GENERATION_RETRY_PROMPT + "\n\nPlease return the JSON now for SQL:\n" + sql_query
                    time.sleep(0.2 * attempt)
                    continue
                else:
                    raise

    # -------- Plan -> SQL prompt and validation --------
    def _generate_plan_to_sql_prompt(self, execution_plan: List[str]) -> Tuple[str, str]:
        """
        Generate a system + user prompt pair that strongly enforces:
        - removal of double-square-bracket tokens ([[...]]) by mapping them to identifiers
        - returning ONLY a single SQLite-compatible SQL statement wrapped in <SQL>...</SQL>
        - high accuracy for filters, joins, aggregates, groupings, ordering, and limits

        Recommendation when calling your LLM: use deterministic sampling (e.g. temperature=0)
        and a token limit large enough for the SQL you expect.
        """
        PLAN_JSON_KEY = "execution_plan"

        system_prompt = (
            "You are a precise translator that converts a natural-language execution plan (provided as JSON) "
            "back into a single, valid SQL query. Follow these rules EXACTLY:\n\n"
            "1) OUTPUT ONLY the SQL query wrapped inside <SQL>...</SQL> tags and NOTHING ELSE — no explanation, "
            "no commentary, and no extra whitespace outside the tags.\n"
            "2) Use standard SQLite-compatible SQL syntax.\n"
            "3) Double-square-bracket tokens (for example [[table]], [[column]], or [[table.column]]) mark identifiers. "
            "You MUST replace them with the identifier text only (no brackets). Under NO CIRCUMSTANCES should the output contain '[' or ']'.\n"
            "4) If an identifier contains characters other than letters, digits, or underscore, wrap it in double quotes (SQLite style). "
            "Escape any internal double quotes by doubling them.\n"
            "5) When both table and column are referenced, prefer table.column (e.g. singer.name). If the plan already uses [[table.column]] then map directly to table.column.\n"
            "6) Implement filters, join keys, aggregations, groupings, ordering, and limits exactly as described by the plan. "
            "If the plan implies combining rows from two tables without specifying join type, use INNER JOIN.\n"
            "7) Produce a single complete SQL statement. Do NOT output multiple statements, comments, or trailing semicolons.\n"
            "8) Place the final SQL on a single line (no leading or trailing newlines) between the tags.\n\n"
            "Accuracy is paramount. If the plan implies an aggregate or grouping, include the correct aggregate functions and GROUP BY expressions."
        )

        # --- Few-shot examples to show exact expected behaviour ---
        example_plan_1 = {
            PLAN_JSON_KEY: [
                "Access the [[singer]] table to retrieve all records.",
                "Count all rows in the [[singer]] table to determine the total number of records."
            ]
        }
        example_sql_1 = "<SQL> SELECT COUNT(*) FROM singer </SQL>"

        example_plan_2 = {
            PLAN_JSON_KEY: [
                "Access the [[albums]] table to retrieve all records.",
                "Keep only rows where [[albums.release_year]] is greater than or equal to 2010 and [[albums.genre]] equals 'rock'.",
                "Return the columns [[albums.id]], [[albums.title]], and [[albums.release_year]] for the remaining rows, ordered by [[albums.release_year]] descending."
            ]
        }
        example_sql_2 = "<SQL> SELECT albums.id, albums.title, albums.release_year FROM albums WHERE albums.release_year >= 2010 AND albums.genre = 'rock' ORDER BY albums.release_year DESC </SQL>"

        example_plan_3 = {
            PLAN_JSON_KEY: [
                "Access the [[singer]] and [[album]] tables.",
                "Keep only rows where [[singer.id]] equals [[album.singer_id]].",
                "For each singer, count the number of matching [[album.id]] rows and call that [[album_count]].",
                "Return [[singer.name]] and [[album_count]] sorted by [[album_count]] descending, and keep only the top 5 rows."
            ]
        }
        example_sql_3 = "<SQL> SELECT singer.name, COUNT(album.id) AS album_count FROM singer JOIN album ON singer.id = album.singer_id GROUP BY singer.name ORDER BY album_count DESC LIMIT 5 </SQL>"

        # Build the JSON for the actual plan to be converted
        plan_json = json.dumps({PLAN_JSON_KEY: execution_plan}, ensure_ascii=False)

        user_prompt = (
            f"Input: a JSON object with key \"{PLAN_JSON_KEY}\" that maps to an array of natural-language steps. "
            "Each step avoids SQL keywords and references tables/columns using double square brackets like [[table]] and [[column]].\n\n"
            "Example input JSON:\n"
            f"{json.dumps(example_plan_1, ensure_ascii=False, indent=2)}\n"
            "Example output (exactly this format):\n"
            f"{example_sql_1}\n\n"
            "Example input JSON:\n"
            f"{json.dumps(example_plan_2, ensure_ascii=False, indent=2)}\n"
            "Example output (exactly this format):\n"
            f"{example_sql_2}\n\n"
            "Example input JSON:\n"
            f"{json.dumps(example_plan_3, ensure_ascii=False, indent=2)}\n"
            "Example output (exactly this format):\n"
            f"{example_sql_3}\n\n"
            "Now convert the following plan JSON into a single SQLite-compatible SQL statement and return ONLY <SQL>... </SQL> on one line (no extra text):\n"
            f"{plan_json}\n\n"
            "IMPORTANT: Do NOT include any '[' or ']' characters in the output. Replace all [[...]] tokens with plain identifiers (table, column, or table.column). "
            "If an identifier contains spaces or special characters, wrap it in double quotes. "
            "Treat content in single quotes inside the plan as string literals and preserve them in the SQL.\n"
            "Remember: output exactly one SQL statement wrapped in <SQL>...</SQL> and nothing else."
        )

        return system_prompt, user_prompt
        
    def _extract_sql_from_response(self, response: str) -> str:
        if not response or not isinstance(response, str):
            return ''
        m = re.search(r'<SQL>(.*?)</SQL>', response, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # fallback: try to find SQL-like substring
        sql_m = re.search(r'(?i)(\bSELECT\b[\s\S]*?;|\bWITH\b[\s\S]*?;)', response)
        if sql_m:
            return sql_m.group(0).strip()
        return response.strip()

    def _plan_to_sql_with_validation(self, execution_plan: List[str]) -> Tuple[str, int]:
        system_prompt, user_prompt = self._generate_plan_to_sql_prompt(execution_plan)

        for attempt in range(1, self.retries + 1):
            response = self._call_llm(user_prompt, system_prompt)
            sql_text = self._extract_sql_from_response(response)

            if not _looks_like_sql(sql_text):
                self.logger.warning(f"Generated SQL failed basic validation on attempt {attempt}: '{sql_text}'")
                if attempt < self.retries:
                    # Re-prompt with a corrective message and explicit example
                    corrective = (
                        "The SQL you produced is invalid or a placeholder. Produce a valid SQL statement using the exact table/column names present in the plan. "
                        "Return ONLY the SQL wrapped in <SQL>...</SQL> with no extra commentary. "
                        "Example: <SQL> SELECT COUNT(*) FROM singer </SQL>"
                    )
                    user_prompt = corrective + "\n\nPlan JSON:\n" + json.dumps({PLAN_JSON_KEY: execution_plan}, ensure_ascii=False)
                    time.sleep(0.2 * attempt)
                    continue
                else:
                    return sql_text if sql_text else '', -1

            # If looks like SQL, return it. Caller will execute/verify against DB.
            return sql_text, 0

        return '', -1

    # -------- Core processing pipeline --------
    def run_experiment(self, max_questions: Optional[int] = None, start_from: Optional[int] = None):
        if start_from is None:
            last = self._get_last_processed_index()
            start_from = last + 1 if last >= 0 else 0

        resume_info = self.get_resume_info()
        if resume_info['has_existing_results']:
            self.logger.info(f"Resuming from index {start_from}. Remaining: {resume_info['remaining_questions']}")
        else:
            self.logger.info(f"Starting fresh from index {start_from}")

        question_ids = list(self.train_data.keys())[start_from:]
        if max_questions:
            question_ids = question_ids[:max_questions]

        total_questions = len(question_ids)
        if total_questions == 0:
            self.logger.info("No questions to process")
            self._print_final_statistics()
            return

        for idx, qid in enumerate(tqdm(question_ids, desc='Processing')):
            try:
                self._process_single_question(qid, idx + 1, total_questions)
            except Exception as e:
                self.logger.error(f"Error processing question {qid}: {e}")
                self.failed_executions += 1
                # Save immediate progress
                self._save_results()
                continue

            # save periodically
            if (idx + 1) % 10 == 0:
                self._save_results()

        # final save
        self._save_results()
        self._print_final_statistics()

    def _process_single_question(self, question_id: str, current_idx: int, total: int):
        data = self.train_data[question_id]
        question = data.get('question') or data.get('query') or ''
        ground_truth = data.get('ground_truth') or data.get('query') or ''
        db_id = data.get('db_id') or data.get('database_id')

        print('\n' + '=' * 80)
        print(f"Question {current_idx}/{total} (ID: {question_id})")
        print(f"DB: {db_id}")
        print(f"Question NL: {question}")
        print(f"Ground-truth SQL: {ground_truth}")

        # Step 1: generate a validated plan
        try:
            execution_plan = self._plan_generation_with_validation(ground_truth)
            print("Generated plan (validated):")
            for i, s in enumerate(execution_plan, 1):
                print(f"  {i}. {s}")
        except Exception as e:
            print(f"❌ Plan generation failed: {e}")
            data['output_plan'] = f"ERROR: {e}"
            data['output_sql_from_plan'] = f"ERROR: {e}"
            data['output_result'] = -1
            data['plan_to_sql_result'] = -1
            self.failed_executions += 1
            return

        # Step 2: plan -> SQL
        try:
            converted_sql, dummy_flag = self._plan_to_sql_with_validation(execution_plan)
            print("Converted SQL from plan:")
            print(converted_sql)
        except Exception as e:
            print(f"❌ Plan->SQL conversion failed: {e}")
            data['output_plan'] = execution_plan
            data['output_sql_from_plan'] = f"ERROR: {e}"
            data['output_result'] = -1
            data['plan_to_sql_result'] = -1
            self.failed_executions += 1
            return

        # Step 3: verify SQL using comparator
        try:
            verification_result = self.sql_comparator.test_sql_with_db_id(converted_sql, ground_truth, db_id)
        except Exception as e:
            self.logger.warning(f"Verification raised exception: {e}")
            verification_result = -1

        # Update stats and record
        self.total_questions += 1
        self.correct_answers += 1  # we consider plan generation success here
        if verification_result == 1:
            self.plan_to_sql_correct += 1
            print("✅ Plan-to-SQL conversion successful on DB check")
        elif verification_result == 0:
            print("❌ Plan-to-SQL conversion produced incorrect results on DB")
        else:
            print("❌ Plan-to-SQL conversion encountered execution error on DB")

        # store outputs
        data['output_plan'] = execution_plan
        data['output_sql_from_plan'] = converted_sql
        data['output_result'] = 1
        data['plan_to_sql_result'] = verification_result

        # progress print
        plan_accuracy = self.correct_answers / self.total_questions if self.total_questions > 0 else 0
        conversion_accuracy = self.plan_to_sql_correct / self.total_questions if self.total_questions > 0 else 0
        print(f"Current plan generation success rate: {plan_accuracy:.3f} ({self.correct_answers}/{self.total_questions})")
        print(f"Current plan-to-sql accuracy: {conversion_accuracy:.3f} ({self.plan_to_sql_correct}/{self.total_questions})")

    def _clean_data_for_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        essential_keys = ['db_id', 'question', 'ground_truth', 'output_plan', 'output_sql_from_plan', 'output_result', 'plan_to_sql_result']
        for qid, qd in data.items():
            cd = {}
            for k in essential_keys:
                if k in qd:
                    cd[k] = qd[k]
            cleaned[qid] = cd
        return cleaned

    def _save_results(self):
        try:
            os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
            cleaned_new = self._clean_data_for_output(self.train_data)
            # Merge carefully: keep existing keys, update with cleaned_new for processed
            merged = dict(self.existing_results)
            merged.update(cleaned_new)
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved results to {self.output_json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _print_final_statistics(self):
        print('\n' + '=' * 80)
        print('SQL2PLAN REFINED EXPERIMENT COMPLETED')
        print('=' * 80)
        print(f"Total Questions Processed: {self.total_questions}")
        print(f"Successful Plan Generations: {self.correct_answers}")
        print(f"Successful Plan-to-SQL Conversions: {self.plan_to_sql_correct}")
        print(f"Failed Executions: {self.failed_executions}")
        if self.total_questions > 0:
            print(f"Plan generation success rate: {self.correct_answers / self.total_questions:.3f}")
            print(f"Plan-to-sql conversion accuracy: {self.plan_to_sql_correct / self.total_questions:.3f}")
        print(f"Results saved to: {self.output_json_path}")


# ------- CLI entrypoint -------
def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None, help='Maximum number of questions to process')
    parser.add_argument('--start', type=int, default=None, help='Start index')
    parser.add_argument('--resume-info', action='store_true', help='Show resume info and exit')
    parser.add_argument('--dataset', type=str, default='Spider_dev', help='Dataset to process')
    args = parser.parse_args()

    try:
        modelBasePath = base_path + '/Process_model/'
        qwen3_8B_path = 'models--Qwen3-8B'
        spider_dev={
        "train_json_path" : base_path + "/Spider_dev/res.json",
        "db_root_path": base_path + "/Spider_dev/database",
        "output_json_path" : base_path + f"/Spider_dev/EXP_sql2plan_{qwen3_8B_path}.json",
        "model" : modelBasePath + qwen3_8B_path,
        "retries" : LLM_RETRIES,
        }
        spider_train={
            "train_json_path" : base_path + "/Spider_train/train.json",
            "db_root_path": base_path + "/Spider_train/database",
            "output_json_path" : base_path + f"/Spider_train/EXP_sql2plan_{qwen3_8B_path}.json",
            "model" : modelBasePath + qwen3_8B_path,
            "retries" : LLM_RETRIES,
        }
        bird_dev={
            "train_json_path" : base_path + "/Bird_dev/dev_res.json",
            "db_root_path": base_path + "/Bird_dev/dev_databases",
            "output_json_path" : base_path + f"/Bird_dev/EXP_sql2plan_{qwen3_8B_path}.json",
            "model" : modelBasePath + qwen3_8B_path,
            "retries" : LLM_RETRIES,
        }
        bird_train={
            "train_json_path" : base_path + "/Bird_train/train.json",
            "db_root_path": base_path + "/Bird_train/train_databases",
            "output_json_path" : base_path + f"/Bird_train/EXP_sql2plan_{qwen3_8B_path}.json",
            "model" : modelBasePath + qwen3_8B_path,
            "retries" : LLM_RETRIES,
        }


        # exp = SQL2PlanExperiment(**spider_train)
        # exp = SQL2PlanExperiment(**bird_train)
        args.dataset = args.dataset.lower()
        if args.dataset == 'spider_dev':
            exp = SQL2PlanExperiment(**spider_dev)
        elif args.dataset == 'spider_train':
            exp = SQL2PlanExperiment(**spider_train)
        elif args.dataset == 'bird_dev':
            exp = SQL2PlanExperiment(**bird_dev)
        elif args.dataset == 'bird_train':
            exp = SQL2PlanExperiment(**bird_train)
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        if args.resume_info:
            ri = exp.get_resume_info()
            print(json.dumps(ri, indent=2, ensure_ascii=False))
            return

        exp.run_experiment(max_questions=args.max, start_from=args.start)

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise


if __name__ == '__main__':
    main()
