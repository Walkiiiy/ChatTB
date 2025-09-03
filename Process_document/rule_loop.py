import os
import sqlite3

from dotenv import load_dotenv

from tqdm import tqdm

from openai import OpenAI

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import asyncio

import json

import copy

import warnings

import re


load_dotenv()

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

class RuleProcesser:
    def __init__(self,
                 load_path,
                 dump_path,
                 db_root_path,
                 tableSchema_path,
                 amendMCPserver_path='Process_document/MCPserver_getAmends.py',
                 testMCPserver_path='Process_document/MCPserver_testSql.py',
                 ruleMCPserver_path='Process_document/MCPserver_getRules.py',
                 ):
        
        self.load_path=load_path
        self.dump_path=dump_path
        self.db_root_path=db_root_path
        
        with open(self.load_path)as f:
            self.evalRes=json.load(f)

        self.testClient=MCPClient(testMCPserver_path)
        self.amendClient=MCPClient(amendMCPserver_path)
        self.ruleClient=MCPClient(ruleMCPserver_path)

        with open(tableSchema_path) as f:
            self.tableSchema=json.load(f)

        self.evalID="0"
    def schema_to_natural_language(self, schema_list):
        """
        Convert schema analysis results into natural language descriptions (English).
        :param schema_list: list of schema dictionaries
        :return: dict {column_name: description}
        """
        descriptions = {}

        for info in schema_list:
            table = info.get("belongsToTable")
            col_name = info.get("originColumnName")
            col_type = info.get("dataFormat")
            size = info.get("size")
            empty = info.get("emptyValueCount")
            val_type = info.get("valType")
            samples = info.get("samples", [])

            avg = info.get("averageValue")
            min_val = info.get("minimumValue")
            max_val = info.get("maximumValue")
            var_val = info.get("sampleVariance")

            fk = info.get("foreignKey")

            parts = []
            parts.append(f"In table **{table}**, there's column **{col_name}** (type: {col_type}).")

            if size:
                parts.append(f"It contains {size} records, with {empty} null values.")

            if val_type:
                parts.append(f"This column mainly represents {val_type}.")

            if avg is not None or min_val is not None or max_val is not None:
                num_desc = []
                if min_val is not None and max_val is not None:
                    num_desc.append(f"the values range from {min_val} to {max_val}")
                if avg is not None:
                    num_desc.append(f"the average value is about {round(avg, 3)}")
                if var_val is not None:
                    num_desc.append(f"the variance is {round(var_val, 3)}")
                if num_desc:
                    parts.append("Statistics show that " + ", ".join(num_desc) + ".")

            if fk:
                parts.append(f"This column is a foreign key, referencing **{fk['toTable']}({fk['toColumn']})**.")

            if samples:
                parts.append(f"Sample values include {samples}.")

            descriptions[col_name] = " ".join(parts)

        return descriptions


    def fetch_linked_schema(self,db_name,scentence):
        # words = scentence.split()
        # words = [word.strip(' ,;\"\'`') for word in words]
        pattern = r"(?:'([^']*)'|\"([^\"]*)\"|`([^`]*)`)"
        matches = re.findall(pattern, scentence)
        # # 每个匹配是一个三元组，只有一个非空，取非空的那个
        words=[m[0] or m[1] or m[2] for m in matches]
        words=set(words)
        schemas=[]
        for word in words:
            if word in self.tableSchema[db_name]:
                schemas.append(self.tableSchema[db_name][word])
        return self.schema_to_natural_language(schemas)
    
    def generate_comment_prompt_getSQLbyAmends(self, question,amends):
        base = "the database's schema is shown above, now you are required to "
        base += "solve the following question related to the database by generating SQLite query."
        base += "the amends contains the latent mistakes you may make, only by avoiding these mistakes can you generate the correct SQLite query." 
        prompt = f"\n{base}\n-- question:\n{question} \n-- amends:\n{amends}"
        return prompt

    def generate_comment_prompt_getSQLbyRules(self, question,rules):
        base = "the database's schema is shown above, now you are required to "
        base += "solve the following question related to the database by generating SQLite query."
        base += "the rules contains the important rules you have to obey, only by obeying these rules can you generate the correct SQLite query." 
        prompt = f"\n{base}\n-- question:\n{question} \n-- rules:\n{rules}"
        return prompt
    
    def generate_comment_prompt_testPlain(self, question):
        base = "the database's schema is shown above, now you are required to "
        base += "solve the following question related to the database by generating SQLite query."
        prompt = f"\n{base}\n-- question:\n{question}"
        return prompt
    
    def generate_schema_prompt(self,db_path, num_rows=None):
        full_schema_prompt_list = []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            if table[0] == 'sqlite_sequence':
                continue
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}';")
            create_prompt = cursor.fetchone()[0]
            schemas[table[0]] = create_prompt
            if num_rows:
                cur_table = f"`{table[0]}`" if table[0] in ['order', 'by', 'group'] else table[0]
                cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_rows}")
                column_names = [description[0] for description in cursor.description]
                values = cursor.fetchall()
                rows_prompt = self.nice_look_table(column_names=column_names, values=values)
                verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
                schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"
        conn.close()
        for v in schemas.values():
            full_schema_prompt_list.append(v)
        return "\n\n".join(full_schema_prompt_list)

    def nice_look_table(self,column_names: list, values: list):
        rows = []
        widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
        header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
        for value in values:
            row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
            rows.append(row)
        return header + '\n' + '\n'.join(rows)

    def evaluate_amendRes(self,evalObj):
        predicted_sql=evalObj['amend_sql'][-1]
        ground_truth=evalObj['ground_truth']
        db_path=os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite')
        try:
            res=self.execute_sql(predicted_sql,ground_truth,db_path)
        except Exception as e:
            print(f"Error SQL in for evalID {self.evalID}: {e}")
            res=0    
        evalObj['amend_res']=res
        return res
    
    def evaluate_ruleRes(self,evalObj):
        print('evaluating rule res...')
        predicted_sql=evalObj['rule_sql'][-1]
        ground_truth=evalObj['ground_truth']
        db_path=os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite')
        try:
            res=self.execute_sql(predicted_sql,ground_truth,db_path)
        except Exception as e:
            print(f"Error SQL in for evalID {self.evalID}: {e}")
            res=0    
        evalObj['rule_res']=res
        return res
    
    def execute_sql(self, predicted_sql, ground_truth, db_path):
        conn = sqlite3.connect(db_path)
        # ⚡ Return text as bytes instead of forcing UTF-8
        conn.text_factory = bytes
        cursor = conn.cursor()
        try:
            cursor.execute(ground_truth)
        except Exception as e:
            print(f"Error executing ground truth SQL: {ground_truth}, {e}")
            return -1
        ground_truth_res = [
            tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
            for row in cursor.fetchall()
        ]
        try:
            cursor.execute(predicted_sql)
            predicted_res = [
                tuple(v.decode("latin1") if isinstance(v, bytes) else v for v in row)
                for row in cursor.fetchall()
            ]
        except Exception as e:
            print(f"Error executing predicted SQL: {predicted_sql}, {e}")
            return 0

        return int(set(predicted_res) == set(ground_truth_res))


    def getPrompt_getSQLbyAmends(self,evalObj):
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries.
            '''
        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        comment_prompt=self.generate_comment_prompt_getSQLbyAmends(evalObj['question'], {'\n'.join(evalObj['amends']) if len(evalObj['amends'])>0 else ''})
        Prompt=f'''
            you will be given database schema, a question related to the database and some amends.
            you should generate a SQLite query that solve the question with the help of amends.
            the amends contains all the latent mistakes you may make while generating the target sql, you need obey all of them.
            you have one function, you have to call it.
            function SQLite_receiver takes the final SQLite query you generate.
            \n-- database schema:\n{schema_prompt},
            \n{comment_prompt}, 
            \nyou have to call a function `SQLite_receiver` to return the SQL you generate'''
        return systemPrompt,Prompt
    
    def getPrompt_getAmends(self,evalObj):
        systemPrompt='''
            You are a precise SQL remediation assistant.
            '''
        prompt='''
            Given a WRONG\_SQL and a RIGHT\_SQL, output exactly one cohesive paragraph (no headings, no lists, no extra lines) composed of “Do … instead of …” sentences so both the amendment and the original are preserved. Use very short backticked snippets (≤4 tokens) to reference elements; never include either full query. Describe changes in this sequence: SELECT list (columns, expressions, aliases, aggregates), FROM sources and join types/ON predicates, WHERE filters, GROUP BY/HAVING, window functions (PARTITION/ORDER), subqueries/CTEs and correlations, ORDER BY/LIMIT/OFFSET, DISTINCT vs UNION/UNION ALL, and any casts/date handling/NULL semantics. For each difference, write a sentence like: “Do `LEFT JOIN` on `t1.id=t2.id` instead of `INNER JOIN` on `t1.id=t2.t1_id`.” If something is added/removed/moved, say “Do add `col_x` instead of omitting it,” “Do remove `DISTINCT` instead of keeping it,” or “Do move filter to `HAVING` instead of `WHERE`.” Call out added/removed tables or conditions, changed join direction, predicate fixes, and why the RIGHT\_SQL logic is correct when it repairs a bug. Do not invent schema and ignore purely cosmetic formatting differences. End with a brief confirmation that the amended query now matches RIGHT\_SQL’s behavior. Inputs: WRONG\_SQL = `{wrong_sql}` RIGHT\_SQL = `{right_sql}`. Output: one paragraph only.
            you should call function amends_receiver the return the output amends paragraph.
            '''
        prompt+=f'''
        the wrong SQL\n:
        {evalObj['amend_sql'][-1] if len(evalObj['amend_sql'])>0 else ''}
        the righi SQL\n:
        {evalObj['ground_truth']}
        '''

        return systemPrompt,prompt
    
    def getPrompt_getSQLbyRules(self,evalObj):
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries.
            '''
        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        comment_prompt=self.generate_comment_prompt_getSQLbyRules(evalObj['question'], {'\n'.join(evalObj['rules']) if len(evalObj['rules'])>0 else ''})
        Prompt=f'''
            you will be given database schema, a question related to the database and some rules.
            you should generate a SQLite query that solve the question with the help of rules.
            the rules contains all the rules you should obey while generating the target sql, you have to obey all of them.
            you have one function, you have to call it.
            function SQLite_receiver takes the final SQLite query you generate.
            \n-- database schema:\n{schema_prompt},
            \n{comment_prompt}, 
            \nyou have to call a function `SQLite_receiver` to return the SQL you generate'''
        return systemPrompt,Prompt
     
    def getPrompt_getRules(self,evalObj,amend_id):
        linked_schema=self.fetch_linked_schema(evalObj['db_id'], evalObj['amends'][-1])
        systemPrompt='''
            You are a helpful assistant that extracts valid rules of a database schmea.
        '''
        prompt=f'''
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (natural-language description of tables, keys, columns).
        3) An incorrect query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect query into the correct query.

        Goal:
        Using ONLY the linked schema and the amends, produce a **single plain-string** numbered list of concise, schema-specific rules. 
        Each rule must be a short **conditional** tied to the question and/or schema (start with “When …:”). 
        Express actions in **natural language only** — do not write SQL keywords. 
        Keep exact tokens for identifiers and literals.

        STRICT OUTPUT FORMAT:
        - Return exactly one plain string: a numbered list like `"1) ... 2) ... 3) ..."`.
        - **Every item MUST start with `When ...:`** (to state the explcit condition of this rule).
        - Include **exact identifier tokens** (tables/aliases/columns with quotes/case) and **exact string literals** where needed.
        - Pay special attention to the output columns' order`
        - After composing the string, call `rules_receiver` with it.

        GLOBAL ACCURACY REQUIREMENTS (encode them as conditional rules; do not use SQL keywords):
        - Dialect & quoting: If schema/amends show identifiers with double quotes, wrap those identifiers in **double quotes** exactly as shown; **never** use backticks.
        - Exact-token mapping: Map phrases from the question to the **exact** column tokens with the correct alias/quotes/spaces (e.g., `county → T1."County Name"`).
        - Literal preservation: When the question or amends specify a text value, match it **exactly** (case, spacing, punctuation), e.g., `'Directly funded'`.
        - Canonical aliases: Use the alias plan implied by the amends (base table = `T1`, first related table = `T2`, etc.); do not swap alias meanings.
        - Join keys/types: When combining tables identified in the amends, match rows using the **exact key equality** shown (e.g., `T1.CDSCode = T2.CDSCode`) and keep only rows present on both sides (inner match). Do not invent alternative keys.
        - Counting: When counting an entity, count using the entity’s **canonical key** shown in the amends (e.g., use `T1.CDSCode` as the counting key), not a generic “all columns”.
        - Operator semantics: Preserve the exact comparison bounds/direction shown in the amends (e.g., inclusive ranges, ≤ / ≥ behavior).

        GENERALIZATION RULE SHAPE (examples you may use; adapt to the amends):
        - `When answering about "<SIMPLIFIED QUESTION>": make sure the output order:<COLUMN>...`
        - `When the question mentions "<X>": "<X>" actually means "<Y> in schema"`
        - `When the question mentions "top", "highest", "largest", or an explicit K by <COLUMN>: rank rows by <COLUMN> in the required direction and keep the first <K> rows.`
        - `When the question asks for a rank range <START_RANK>.. <END_RANK> after ranking on <COLUMN>: take rows from position <START_RANK> through <END_RANK> inclusive (where start is position 1).`
        - `When the question says "per <GROUP>"/"by <GROUP>": organize results by the token for <GROUP> and compute requested aggregates per group.`
        - `When the question asks for "count of <ENTITY>": compute the number of rows using the canonical key token for <ENTITY> (e.g., T1.CDSCode).`
        - `When the question asks for "distinct <X>": compute the number of **unique** values of the exact token for <X>.`
        - `When the question asks for a ratio "A to B": compute (rows satisfying <COND_A>) divided by (rows satisfying <COND_B>), guarding division by zero; express conditions using exact tokens and literals from the amends.`
        - `When combining T1 and T2 for shared entities: link rows where the canonical keys are equal exactly as shown in the amends (e.g., T1.CDSCode = T2.CDSCode); keep only matching pairs (inner match).`
        - `When the question implies ordering ties: break ties using the canonical key if shown in the amends (e.g., T1.CDSCode).`

        FORBIDDEN TRANSFORMS (write as conditional negatives; no SQL keywords):
        - `When choosing identifier delimiters: do not replace double quotes with backticks or unquoted forms.`
        - `When handling text literals: do not change case, spacing, or punctuation.`
        - `When referring to table roles: do not rename or swap the aliases T1, T2, ... once set for this schema.`
        - `When combining tables specified as an inner match in the amends: do not perform outer matches or cartesian combinations.`
        - `When producing output or grouping: do not reorder columns or grouping keys beyond what the amends and question require.`

        PLACEHOLDERS (allowed; keep them minimal and plain-language):
        - `<K>`, `<COLUMN>`, `<START_RANK>`, `<END_RANK>`, `<GROUP>`, `<COND_A>`, `<COND_B>`.

        Inputs for grounding:
        the question:
        {evalObj['question']}

        the wrong query:
        {evalObj['amend_sql'][amend_id]}

        the amends:
        {evalObj['amends'][amend_id]}

        the linked schema:
        {linked_schema}

        Output:
        Return the numbered list as one plain string, then call `rules_receiver` with that string.
        '''
        promptbak9=f'''
        SCHEMA-SPECIFIC, CONDITIONAL RULES — INTENTION → ACTION (NO SQL OUTPUT)

        You are given five inputs:
        1) QUESTION (natural language)
        2) SCHEMA (natural-language description of tables, keys, columns, types, PK/FK, grain)
        3) WRONG_SQL (incorrect query)
        4) RIGHT_SQL (correct query)
        5) AMENDS (a bullet list of edits that transform WRONG_SQL into RIGHT_SQL)

        GOAL
        Using ONLY SCHEMA and AMENDS (you may consult RIGHT_SQL/WRONG_SQL just to understand failure reasons), produce a SINGLE PLAIN STRING that contains a numbered list of concise, schema-specific **conditional rules** the system can reuse for similar tasks. Each rule must capture:
        • the INTENTION (what the user or query is trying to do),
        • the KEYWORDS that signal that intention,
        • the precise ACTION tied to the SCHEMA (what to select/filter/compute/rank/join/aggregate/cast/etc).

        STYLE & FORMAT (STRICT)
        * Treat SQL keywords as forbidden vocabulary (except when they appear inside official column/table names).
        * Each item must start with: `When <Intention> <Keywords>:` followed by the ACTION.
        * Use **exact schema tokens** (table and column names exactly as in SCHEMA).
        * Do **not** wrap schema tokens in quotes/backticks unless:
        • the schema token itself contains spaces, punctuation, or special characters, or
        • the token appears inside an inline expression (e.g., ``(`numerator` / `denominator`)``).
        * Do not output SQL code or keywords like SELECT/JOIN/GROUP BY; describe actions in natural language.
        * Be schema-specific, not generic. No invented columns/tables.
        * Encode conditions that explain **when** to use an action, not just the action itself.
        * Rules must be **atomic** (one intention → one action). If AMENDS implies multiple independent behaviors, make multiple rules.
        * Avoid duplication; merge overlapping rules by generalizing their condition minimally while keeping them schema-true.
        * Keep each rule ≤ 30 words where possible.
        * OUTPUT MUST BE A SINGLE STRING with items numbered `1)`, `2)`, `3)` … **on one line**, separated by a single space. No extra prose before/after.

        THINKING STEPS (DO NOT OUTPUT THESE; USE INTERNALLY)
        1) Map INTENTS: From QUESTION + AMENDS + RIGHT_SQL, list the concrete intents (e.g., “find highest rate”, “filter by county”, “use correct table”).
        2) Keyword cues: For each intent, extract natural-language cues that would trigger it (e.g., “highest/top/max”, “in Alameda County”, “K-12 only”, “rate/percentage/ratio”).
        3) Failure diagnosis: Contrast WRONG_SQL with AMENDS/RIGHT_SQL to identify why it failed (wrong table, wrong column, premature aggregate, missing ORDER BY/LIMIT, integer division, wrong filter column/value, wrong grain, missing join).
        4) Schema anchoring: For each diagnosed fix, tie the action to exact SCHEMA tokens and correct grain (row vs aggregate) and note type/casting if needed.
        5) Generalize minimally: Turn each fix into a **When …: <Action>** rule that will hold for similar queries in this SCHEMA (not across schemas). Prefer rank-then-pick over aggregates when the task is “top/bottom N with attributes”.
        6) Validate: No invented tokens; every token must appear in SCHEMA. If AMENDS referenced a precomputed column that conflicts with a more accurate computation, encode a rule that prefers the explicit computation and states when.
        7) Order rules from most to least central to answering the QUESTION.

        OUTPUT TEMPLATE
        Return exactly one string like:
        `1) When <intention> <keywords>: <action>. 2) When <intention> <keywords>: <action>. 3) ...`

        EXAMPLES OF INTENTION PHRASES (USE ONLY IF THEY MATCH THE INPUTS)
        • calculating | computing | deriving a rate/ratio/percentage
        • selecting | filtering by <dimension/value> (e.g., County Name, Educational Option Type)
        • ranking | finding highest/lowest/top/bottom N
        • picking a source table when multiple contain overlapping columns
        • joining entities to reach required columns via PK/FK
        • handling numeric division/casting to avoid integer truncation
        • choosing row-grain ranking + limit vs. aggregates that drop attributes
        • case-sensitive value matching as specified by SCHEMA
        • deduplicating vs counting distinct when grain requires it
        • time-range or category scoping (e.g., K-12 subset) exactly as per SCHEMA        
        INPUTS
        QUESTION:
        {evalObj['question']}

        SCHEMA:
        {linked_schema}

        WRONG_SQL:
        {evalObj['amend_sql'][amend_id]}

        RIGHT_SQL:
        {evalObj['ground_truth']}

        AMENDS:
        {evalObj['amends'][amend_id]}
        '''
        promptbak8=f'''
        CONDITIONAL, DB-AGNOSTIC RULES — NO SQL KEYWORDS
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (natural-language description of tables, keys, columns).
        3) An incorrect query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect query into the correct query.

        Goal:
        Using ONLY the linked schema and the amends, produce a **single plain-string** numbered list of concise, schema-specific rules. 
        Each rule must be a short **conditional** tied to the question and/or schema (start with “When …:”). 
        Express actions in **natural language only** — do not write SQL keywords. 
        Keep exact tokens for identifiers and literals.

        STRICT OUTPUT FORMAT:
        - Return exactly one plain string: a numbered list like `"1) ... 2) ... 3) ..."`.
        - **Every item MUST start with `When ...:`** (no “Always” rules).
        - Include **exact identifier tokens** (tables/aliases/columns with quotes/case) and **exact string literals** where needed.
        - Include **exactly one rule titled `Output columns (ordered):`** listing the identifiers in the required order, e.g.:  
            `Output columns (ordered): T2.Website, T1."School Name"`
        - After composing the string, call `rules_receiver` with it.

        GLOBAL ACCURACY REQUIREMENTS (encode them as conditional rules; do not use SQL keywords):
        - Dialect & quoting: If schema/amends show identifiers with double quotes, wrap those identifiers in **double quotes** exactly as shown; **never** use backticks.
        - Exact-token mapping: Map phrases from the question to the **exact** column tokens with the correct alias/quotes/spaces (e.g., `county → T1."County Name"`).
        - Literal preservation: When the question or amends specify a text value, match it **exactly** (case, spacing, punctuation), e.g., `'Directly funded'`.
        - Canonical aliases: Use the alias plan implied by the amends (base table = `T1`, first related table = `T2`, etc.); do not swap alias meanings.
        - Join keys/types: When combining tables identified in the amends, match rows using the **exact key equality** shown (e.g., `T1.CDSCode = T2.CDSCode`) and keep only rows present on both sides (inner match). Do not invent alternative keys.
        - Counting: When counting an entity, count using the entity’s **canonical key** shown in the amends (e.g., use `T1.CDSCode` as the counting key), not a generic “all columns”.
        - Operator semantics: Preserve the exact comparison bounds/direction shown in the amends (e.g., inclusive ranges, ≤ / ≥ behavior).

        GENERALIZATION RULE SHAPE (examples you may use; adapt to the amends):
        - `When the question mentions "top", "highest", "largest", or an explicit K by <COLUMN>: rank rows by <COLUMN> in the required direction and keep the first <K> rows.`
        - `When the question asks for a rank range <START_RANK>.. <END_RANK> after ranking on <COLUMN>: take rows from position <START_RANK> through <END_RANK> inclusive (where start is position 1).`
        - `When the question says "per <GROUP>"/"by <GROUP>": organize results by the token for <GROUP> and compute requested aggregates per group.`
        - `When the question asks for "count of <ENTITY>": compute the number of rows using the canonical key token for <ENTITY> (e.g., T1.CDSCode).`
        - `When the question asks for "distinct <X>": compute the number of **unique** values of the exact token for <X>.`
        - `When the question asks for a ratio "A to B": compute (rows satisfying <COND_A>) divided by (rows satisfying <COND_B>), guarding division by zero; express conditions using exact tokens and literals from the amends.`
        - `When combining T1 and T2 for shared entities: link rows where the canonical keys are equal exactly as shown in the amends (e.g., T1.CDSCode = T2.CDSCode); keep only matching pairs (inner match).`
        - `When the question implies ordering ties: break ties using the canonical key if shown in the amends (e.g., T1.CDSCode).`

        FORBIDDEN TRANSFORMS (write as conditional negatives; no SQL keywords):
        - `When choosing identifier delimiters: do not replace double quotes with backticks or unquoted forms.`
        - `When handling text literals: do not change case, spacing, or punctuation.`
        - `When referring to table roles: do not rename or swap the aliases T1, T2, ... once set for this schema.`
        - `When combining tables specified as an inner match in the amends: do not perform outer matches or cartesian combinations.`
        - `When producing output or grouping: do not reorder columns or grouping keys beyond what the amends and question require.`

        PLACEHOLDERS (allowed; keep them minimal and plain-language):
        - `<K>`, `<COLUMN>`, `<START_RANK>`, `<END_RANK>`, `<GROUP>`, `<COND_A>`, `<COND_B>`.

        Inputs for grounding:
        the question:
        {evalObj['question']}

        the wrong query:
        {evalObj['amend_sql'][amend_id]}

        the amends:
        {evalObj['amends'][amend_id]}

        the linked schema:
        {linked_schema}

        Output:
        Return the numbered list as one plain string, then call `rules_receiver` with that string.
        '''
        promptbak7=f'''
        SIMPLE REFINED PROMPT — Add short conditional, schema+question rules

        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (natural-language description of tables, keys, columns).
        3) An incorrect SQL query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect SQL into the correct SQL.

        Task:
        Using ONLY the linked schema and the amends, produce a single plain-string numbered list of concise, schema-specific rules that a template generator can follow. Rules must be short, token-accurate, and include simple conditions that combine the question text and schema/amend tokens (e.g., "When question asks...", "When joining...").

        MANDATORY OUTPUT FORMAT:
        - Return exactly one plain string: a numbered list like:  
            `"1) ... 2) ... 3) ..."`
        - Each rule MUST begin with either `When <condition>:` or `Always:`. Example conditions: `When question asks "top" or "highest":`, `When joining T1 and T2:`, `When selecting a rank range:`.
        - Include exactly one rule titled `Select-list (ordered):` containing the precise SELECT clause tokens **in order**, e.g.:  
            `Select-list (ordered): SELECT T2.Website, T1."School Name"`
        - After composing the string, call `rules_receiver` with it.

        ESSENTIAL ACCURACY RULES (must be encoded as short rules):
        - A) **Dialect & quoting**: Always use the quoting shown in schema/amends (e.g., `T1."County Name"`). NEVER use backticks if double-quotes appear. Use single quotes for string literals (e.g., `'Directly funded'`).
        - B) **Exact-token mapping**: Map natural-language phrases to exact tokens. Provide an exact token rule and a conditional mapping:  
            `When question contains "county": use T1."County Name"`.
        - C) **Literal preservation**: Preserve string literal case/spacing exactly: `T1."Charter Funding Type" = 'Directly funded'`.
        - D) **Canonical aliases**: `Always: base=T1, first join=T2, ...` (use aliases from amends).
        - E) **Join keys & types**: `When joining T1 and T2: use INNER JOIN T2 ON T1.CDSCode = T2.CDSCode` (exact tokens from amends).
        - F) **Count/aggregate**: `When asking "count of X": use COUNT(T1.<X-key>)` (exact token).
        - G) **Selection discipline**: `Always: select only columns required by question and amends` (enforce `Select-list (ordered)`).
        - H) **Numeric/logic operators**: `When question implies bounds "between", "<=", ">=": preserve operator/inclusivity exactly as in amends`.

        SIMPLE GENERALIZATION FOR CONDITIONS (short allowed patterns):
        - `When question contains any of: "top","highest","rank","nth": ORDER BY <COLUMN> <SORT_DIR> LIMIT <K>` (you may use `<K>` placeholder).
        - `When selecting rank range <START_RANK>.. <END_RANK>: use LIMIT <OFFSET>, <COUNT>` and show formula: `<OFFSET> = <START_RANK> - 1; <COUNT> = <END_RANK> - <START_RANK> + 1`. (Only include constants when the amends mark them canonical.)
        - `When joining T1 and T2: use the exact ON-condition tokens shown in amends.`

        FORBIDDEN TRANSFORMS (must be stated):
        - Do not change identifier quoting style (e.g., NEVER replace `T1."County Name"`).
        - Do not normalize case of string literals or column names.
        - Do not rename or swap table aliases once assigned.
        - Do not use implicit joins if amends use explicit `INNER JOIN`.
        - Do not reorder the SELECT list or GROUP BY columns.

        Inputs for grounding:
        the question: {evalObj['question']}
        the wrong SQL: {evalObj['amend_sql'][amend_id]}
        the amends: {evalObj['amends'][amend_id]}
        the linked schema: {linked_schema}

        Output:
        Call `rules_receiver` with the single-string numbered list of short, conditional, schema+question rules.
        '''
        promptbak6=f'''
        REFINED PROMPT — Produce generalized, schema-level rule templates from amends
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (a natural-language description of tables, keys, columns).
        3) An incorrect SQL query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect SQL into the correct SQL.

        Task:
        Using ONLY the linked schema and the amends, distill a concise, enforceable set of **schema-level rules and parameterized patterns** that SQL generation for THIS schema must follow so future queries avoid the same errors. Produce rules that:
        - are concrete and token-level where needed (so a template generator can emit tokens),
        - but are generalized (use placeholders/formulas rather than embedding per-question constants when appropriate),
        - and preserve any example-specific exact tokens from the amends when they are canonical.

        Key principles you MUST follow in rule creation:
        - Keep exact-token fidelity for schema identifiers and string literals (see policies A–C below).
        - When an amend shows a pattern (e.g., pagination, counting, nth-highest), output a **parameterized template** (use placeholders like `<OFFSET>`, `<COUNT>`, `<START_RANK>`, `<END_RANK>`, `<COLUMN>`, `<SORT_DIR>`). Also show the exact formula that converts natural-language parameters into those placeholders (e.g., `<OFFSET> = <START_RANK> - 1`, `<COUNT> = <END_RANK> - <START_RANK> + 1`).
        - For each canonical token discovered in the amends (column tokens, join keys, aliases), produce BOTH:
            1) an exact-token rule (e.g., `county → T1."County Name"`), and
            2) a generalized mapping template that other questions can reuse (e.g., `the word "county" in the question → use token T1."County Name"`).
        - Group similar amend types into single rules where possible (e.g., all pagination amends → one pagination rule template).
        - Preserve mandatory constraints from the amends exactly (SELECT-list order, quoting, canonical aliases, join conditions, COUNT semantics, etc.). Do **not** generalize away constraints that are explicitly canonical in the amends.

        Critical policy tokens that MUST appear as explicit tokens/rules (these are mandatory):
        A) **Dialect & quoting**: If identifiers in the schema/amends use double quotes anywhere, ALWAYS use double quotes for those identifiers (e.g., `T1."County Name"`). NEVER use backticks if amends show double-quote style. Always use single quotes for string literals (e.g., `'Directly funded'`).
        B) **Exact-token mapping**: Map every referenced natural-language phrase to the exact `T."Column Name"` token with correct table alias and exact spelling/spaces/quotes. Example: `county → T1."County Name"`.
        C) **Literal preservation**: For text equality filters, copy literal values EXACTLY (case, spacing, punctuation) as in amends: `T1."Charter Funding Type" = 'Directly funded'`.
        D) **Canonical aliases**: Fix canonical aliases: base table = `T1`, first joined table = `T2`, etc., as implied by amends.
        E) **Join keys**: Use the exact join condition(s) from the amends as canonical (e.g., `T1.CDSCode = T2.CDSCode`) and the exact join type (`INNER JOIN`). Do not invent alternative keys or join types.
        F) **Count/aggregate semantics**: When counting entities, use the canonical key from the amends (e.g., `COUNT(T1.CDSCode)`) not `COUNT(*)`, unless amends state otherwise.
        G) **Selection discipline**: Only select columns required by the question AND the amends.
        H) **Numeric/logic operators**: Preserve operator direction and inclusivity exactly as in amends (e.g., `BETWEEN 1900 AND 2000`, `<= 250`).
        I) **Projection order (MANDATORY)**: If the amends specify selected columns, include a single rule titled exactly `Select-list (ordered)` that contains the precise SELECT clause tokens **in order**, e.g.:  
            `Select-list (ordered): SELECT T2.Website, T1."School Name"`

        NEW — Generalization & placeholder requirements (must be followed):
        J) **Placeholders and formulas**: When a rule would otherwise embed a question-specific constant, replace it with a placeholder and provide an exact formula. Use angle-bracket placeholders such as `<START_RANK>`, `<END_RANK>`, `<OFFSET>`, `<COUNT>`, `<K>`, `<COLUMN>`, `<SORT_DIR>`, `<TABLE_ALIAS>`. Example pagination rule token:  
            `Pagination (rank range): For requested rank range <START_RANK>.. <END_RANK>, use: ORDER BY <COLUMN> <SORT_DIR> LIMIT <OFFSET>, <COUNT>`  
            and include formula tokens: `<OFFSET> = <START_RANK> - 1`, `<COUNT> = <END_RANK> - <START_RANK> + 1`.
        K) **Phrase → meaning patterns**: Whenever an amend reveals a mapping between a natural-language phrase and a semantic intent, produce a rule of the form:  
            `"the word '<phrase>' in question means: <token/template>"`  
            Example: `the phrase "count of schools" means: COUNT(T1.CDSCode)` or `the phrase "distinct X" means: COUNT(DISTINCT <token>)`.
        L) **Aggregation templates**: Provide templates for common aggregates shown in amends: e.g., `Top-K by <COLUMN> → ORDER BY <COLUMN> DESC LIMIT <K>`, `COUNT distinct entity → COUNT(DISTINCT <token>)`, `ratio of A to B → SUM(CASE WHEN <cond_for_A> THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN <cond_for_B> THEN 1 ELSE 0 END),0)`.
        M) **Join-pattern templates**: If amends show repeated join behavior between two tables, output a template: e.g., `When joining T1 and T2 use: INNER JOIN T2 ON T1.CDSCode = T2.CDSCode`. If the join must include filter pushdown or ON-condition filters, show them as exact tokens.
        N) **ORDER BY & tie-breaking**: If amends specify an ORDER BY used for ranking/pagination, encode: `ORDER BY <token(s)> <dir> [ , <tie-breaker-token> ]` and prefer tie-breakers if amends show them.

        Forbidden transformations (must be stated as short rules):
        - Do not change identifier quoting style. (e.g., NEVER replace `T1."County Name"` with `T1.County_Name`).
        - Do not normalize the case of string literals or column names.
        - Do not rename or swap table aliases once assigned.
        - Do not use implicit joins if the amends specify `INNER JOIN` (or an explicit join type).
        - Do not reorder the SELECT list or GROUP BY columns. No heuristic reordering.

        Hard constraints on your OUTPUT (unchanged):
        - Output must be a single plain string: a numbered list of short, imperative, schema-specific rules.
        - Each rule MUST show the exact token(s) where helpful (including quotes and case), not just prose.
        - **Include exactly one rule titled `Select-list (ordered)` that contains the precise SELECT clause tokens in order**, e.g.:  
            `Select-list (ordered): SELECT T2.Website, T1."School Name"`
        - If you use placeholders, show the exact placeholder name and any formula that maps question parameters to placeholder values.
        - Do not output explanations, rationale, or chain-of-thought beyond the numbered rules.
        - Do not split the select-list across multiple rules; keep it in that one ordered rule.

        Formatting & final action:
        - Return the rules as a single numbered list string in this exact format:  
            `"1) ... 2) ... 3) ..."`  
            Keep items short, imperative, and token-rich.
        - After composing that string, call function `rules_receiver` with the string of rules.

        Inputs for grounding:
        the question: {evalObj['question']}
        the linked schema: {linked_schema}
        the wrong SQL: {evalObj['amend_sql'][amend_id]}
        the amends: {evalObj['amends'][amend_id]}

        Output:
        Call `rules_receiver` with the single-string, numbered list of generalized, schema-level rules (following all constraints above).
        '''
        promptbak5=f'''
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (a natural-language description of tables, keys, columns).
        3) An incorrect SQL query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect SQL into the correct SQL.

        Your task:
        - Using ONLY the linked schema and the amends, distill a concise, enforceable set of rules that SQL generation for THIS schema must follow so future queries avoid the same errors.
        - The rules must be concrete enough that a template-based generator could produce the correct SQL tokens without guessing.

        Critical policy to encode (must appear in the rules as explicit tokens):
        A) **Dialect & quoting**: Detect identifier quoting from the linked schema/amends. If identifiers in the schema/amends use double quotes anywhere for identifiers with spaces/case, use double quotes for those identifiers exactly as needed (e.g., T1."County Name"). NEVER use backticks if the schema/amends indicate double-quote style. Always use single quotes for string literals (e.g., 'Directly funded').
        B) **Exact-token mapping**: Map every referenced natural-language phrase to the exact `T."Column Name"` token with correct table alias, spelling, spaces, and quoting taken from the schema/amends. Do not use generic names. Example style: `county → T1."County Name"`, `charter funding type → T1."Charter Funding Type"`, `school id → T1.CDSCode`.
        C) **Literal preservation**: For text equality filters, copy literal values EXACTLY (case, spacing, punctuation) as shown in the schema/amends. Do not lowercase or title-case them. Example: `T1."Charter Funding Type" = 'Directly funded'`.
        D) **Canonical aliases**: Fix a canonical alias plan: base table = T1 (as implied by amends), first joined table = T2, etc. Do not swap them across rules for this schema.
        E) **Join keys**: Specify the exact join condition(s) from the amends as canonical (e.g., `T1.CDSCode = T2.CDSCode`) and the exact join type (e.g., `INNER JOIN`). Do not invent alternative keys or types.
        F) **Count/aggregate semantics**: When counting entities, count the schema’s canonical key for that entity as shown in the amends (e.g., `COUNT(T1.CDSCode)`), not `COUNT(*)`, unless the amends specify otherwise.
        G) **Selection discipline**: Only select columns required by the question AND amends.
        H) **Numeric/logic operators**: Preserve operator direction and inclusivity exactly as in amends (e.g., `BETWEEN 1900 AND 2000`, `<= 250`).
        I) **Projection order (MANDATORY)**: If the amends specify the selected columns, encode a single rule that fixes the exact SELECT list **in order** as a comma-separated token sequence (e.g., `SELECT T2.Website, T1."School Name"`). If the amends don’t specify order, mirror the column order exactly as first stated in the amends; only if the amends are silent, mirror the order in the question.

        Forbidden transformations (must be stated as rules):
        - Do not change identifier quoting style.
        - Do not normalize the case of string literals or column names.
        - Do not rename or swap table aliases once assigned.
        - Do not use implicit joins if the amends specify `INNER JOIN` (or another explicit type).
        - **Do not reorder the SELECT list or GROUP BY columns.** No alphabetical sorting, no “name-first” conventions, no heuristic reordering.

        Hard constraints on your OUTPUT:
        - Output must be a single plain string: a numbered list of short, imperative, schema-specific rules.
        - Each rule MUST show the exact token(s) where helpful (including quotes and case), not just prose.
        - **Include exactly one rule titled “Select-list (ordered)” that contains the precise SELECT clause tokens in order, e.g.: `Select-list (ordered): SELECT T2.Website, T1."School Name"`**
        - Do not split the select-list across multiple rules; keep it in that one ordered rule.
        - Do not output explanations, rationale, or thinking.

        Format:
        Return the rules as a numbered list string, like:
        "1) ... 2) ... 3) ..."

        Input to ground your reasoning:
        the question:
        {evalObj['question']}

        the wrong SQL:
        {evalObj['amend_sql'][amend_id]}

        the amends:
        {evalObj['amends'][amend_id]}

        the linked schema:
        {linked_schema}

        Output:
        Call function rules_receiver with the string of rules.
        '''
        promptbak4=f'''
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (a natural-language description of tables, keys, columns).
        3) An incorrect SQL query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect SQL into the correct SQL.

        Your task:
        - Using ONLY the linked schema and the amends, distill a concise, enforceable set of rules that SQL generation for THIS schema must follow so future queries avoid the same errors.
        - The rules must be concrete enough that a template-based generator could produce the correct SQL tokens without guessing.

        Critical policy to encode (must appear in the rules as explicit tokens):
        A) **Dialect & quoting**: Detect the identifier quoting style from the linked schema/amends. If identifiers in the schema/amends use double quotes, ALWAYS use double quotes for identifiers (e.g., T1."County Name"). NEVER use backticks in that case. Always use single quotes for string literals (e.g., 'Directly funded').
        B) **Exact-token mapping**: Map every referenced natural-language phrase to the exact `T."Column Name"` token with correct table alias, spelling, spaces, and quoting taken from the schema/amends. Do not use generic names. Example style: `county → T1."County Name"`, `charter funding type → T1."Charter Funding Type"`, `school id → T1.CDSCode`.
        C) **Literal preservation**: For text equality filters, copy literal values EXACTLY (case, spacing, punctuation) as shown in the schema/amends. Do not lowercase or title-case them. Example style: `T1."Charter Funding Type" = 'Directly funded'` (NOT 'directly funded').
        D) **Canonical aliases**: Fix a canonical alias plan: base table = T1 (as implied by amends), first joined table = T2, etc. Do not swap them across rules for this schema.
        E) **Join keys**: Specify the exact join condition(s) from the amends as canonical (e.g., `T1.CDSCode = T2.cds`). Do not invent alternative keys.
        F) **Count/aggregate semantics**: When counting entities, count the schema’s canonical key for that entity as shown in the amends (e.g., `COUNT(T1.CDSCode)`), not `COUNT(*)`, unless the amends specify otherwise.
        G) **Selection discipline**: Only select columns required by the question.
        H) **Numeric/logic operators**: Preserve operator direction and inclusivity exactly as in amends (e.g., `<= 250`).

        Forbidden transformations (must be stated as rules):
        - Do not change identifier quoting style.
        - Do not normalize the case of string literals or column names.
        - Do not rename or swap table aliases once assigned.
        - Do not use implicit joins if the amends specify `INNER JOIN` (or another explicit type).

        Hard constraints on your OUTPUT:
        - Output must be a single plain string: a numbered list of short, imperative, schema-specific rules.
        - Each rule MUST show the exact token(s) where helpful (including quotes and case), not just prose.
        - Do not output explanations, rationale, or thinking.

        Format:
        Return the rules as a numbered list string, like:
        "1) ... 2) ... 3) ..."

        Input to ground your reasoning:
        the question:
        {evalObj['question']}

        the wrong SQL:
        {evalObj['amend_sql'][amend_id]}

        the amends:
        {evalObj['amends'][amend_id]}

        the linked schema:
        {linked_schema}

        Output:
        Call function rules_receiver with the string of rules.
        '''
        promptbak3=f'''
        You are given:
        1) A natural-language question about a database schema.
        2) The linked schema (a natural-language description of tables, keys, columns).
        3) An incorrect SQL query generated for the question.
        4) A list of amendments (“amends”) that transform the incorrect SQL into the correct SQL.

        Your task:
        - Using the linked schema and the amends, explain why the amends work and what general mistakes they fix.
        - Then output a concise set of enforceable rules that SQL generation for THIS schema must follow so that future queries avoid the same errors.

        Important method (think silently, do not output reasoning):
        1) Identify the BASE TABLE (the one providing most SELECT fields).
        2) Fix a canonical alias plan: always use T1 for the base table, T2 for the joined table, etc. Do not swap them.
        3) Map question phrases to exact table.column names, with correct spelling and quoting from the schema.
        4) If the question asks for a “difference” or “gap” between numbers, use ABS(A-B) unless the question says otherwise.
        5) Use the exact join keys that appear in the amends as the canonical join condition.
        6) Only select columns explicitly asked in the question; do not add extras.

        Hard constraints:
        - Output must be a single plain string.
        - Rules must be short, imperative, and schema-specific.
        - Do not output explanations, only the rules.

        Format:
        Return the rules as a numbered list string, like:
        "1) ... 2) ... 3) ..."

        Input to ground your reasoning:
        the question:
        {evalObj['question']}

        the wrong SQL:
        {evalObj['amend_sql'][amend_id]}

        the amends:
        {evalObj['amends'][amend_id]}

        the linked schema:
        {linked_schema}

        Output:
        Call function rules_receiver with the string of rules.
        '''
        promptbak2='''
         You are given:
        1. A natural language question about a database schema.
        2. the linked schema of the database, which is a natural language description of the schema.
        3. The incorrect sql query generated by a model in response to the question.
        4. A list of amendments (amends) that explain how to fix an incorrect SQL query into the correct SQL query.
        Your task:
        - Combining the linked schema and the amends, analyze why does the amends work.
        - Answer: what to do to make the future questions dont make the same mistakes again?
        - Return your answer as rules should obey while generating SQL queries of this schema.
        Hard constraints:
        - The rules should be short, concise, and generalized.
        - Call function `rules_receiver` to return the rules.
        '''
        promptbak=f'''
        You are given:
        1. A natural language question about a database schema.
        2. The incorrect sql query generated by a model in response to the question.
        3. The correct sql that solves the question.
        4. A list of amendments (amends) that explain how to fix an incorrect SQL query into the correct SQL query.
        5. the linked schema of the database, which is a natural language description of the schema.
        Your task:
        - Extract **general rules (rules)** from the amends.
        - Classify each rule into one of the following categories:
        1. Ambiguity in question: Keywords in the question do not clearly map to schema metadata, or the question has multiple possible meanings.
        2. Lack of background knowledge: The question contains unknown domain-specific terms or metrics whose calculation is non-standard.
        3. Schema structural issues: Mistakes caused by schema design, such as uneven table sizes, special foreign keys, or join structures.
        4. Output formatting issues: The logical answer is correct, but the returned result format does not match the expectation.
        5. Reasoning/Planning issues: The overall reasoning or strategy is wrong, and the query must be restructured with a new approach.

        Output format:
        - Return one paragraph of listed extracted rules.  
        - Each rule should include:
        - A short, generalized statement of the rule.

        Example:

        Input amends:
        "Do use a single query with conditional aggregation instead of separate CTEs for counts. Do calculate the ratio directly using SUM(CASE...) instead of separate count columns. Do add the filter StatusType = 'Merged' instead of omitting it. Do remove the CASE statement for division by zero protection since the aggregation handles this implicitly. Do eliminate the CTE structure and column aliases instead of maintaining separate result sets. Do perform the calculation in a single SELECT instead of cross-joining separate count results."

        Question:
        "What is the ratio of merged Unified School District schools in Orange County to merged Elementary School District schools?"

        Expected output rules:
        1.Prefer conditional aggregation (SUM with CASE) instead of separate CTE counts. → Category: Reasoning/Planning issues.
        2.Always include domain-specific filters mentioned in the question (e.g., StatusType = 'Merged'). → Category: Ambiguity in question.
        3.Avoid unnecessary division-by-zero handling if aggregation already prevents it. → Category: Reasoning/Planning issues.
        4.Eliminate redundant CTE structures and compute results in a single query. → Category: Schema structural issues.
        5.Use direct calculation in one SELECT rather than cross-joining separate counts. → Category: Reasoning/Planning issues.
        '''
        # prompt+=f'''
        # \nthe question:\n{evalObj['question']}
        # \nthe wrong SQL:\n{evalObj['amend_sql'][amend_id]}
        # \nthe amends:\n{evalObj['amends'][amend_id]}
        # \nthe linked schema:\n{linked_schema}
        # '''
        # prompt+='''
        # \nyou have to call a function `rules_receiver` to return one paragraph of rules you generate.
        # '''
        return systemPrompt, prompt
    
    def getRules(self,evalObj):
        for amend_id,amend in enumerate(evalObj['amends']):
            print(f"getting rules {amend_id}...")
            systemPrompt,prompt=self.getPrompt_getRules(evalObj,amend_id)
            res=self.ruleClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
            if 'error' in res:
                evalObj['rules'].append('error')
            # print(res)        
            evalObj['rules'].append(res['rules'])
    
    def getSQLbyRules(self,evalObj):
        print("testing rules...")
        systemPrompt,prompt=self.getPrompt_getSQLbyRules(evalObj)
        res=self.testClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            evalObj['rule_sql'].append('error')
            return res 
        evalObj['rule_sql'].append(res['sql'])
        return res
    
    def getWrongSQL(self,evalObj):
        print("getting...")
        systemPrompt='you are a helpfule sql generator'
        prompt=f'''
        please generate a wrong sql query according to the following question:
        the question:
        {evalObj['question']}
        \nyou have to call a function `SQLite_receiver` to return the SQL you generate'''
        res=self.testClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            evalObj['amend_sql'].append('error')
            return res 
        evalObj['amend_sql'].append(res['sql'])
        return res
    
    def getAmends(self,evalObj):
        systemPrompt,prompt=self.getPrompt_getAmends(evalObj)
        res=self.amendClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            return res    
        # print(res)        
        evalObj['amends'].append(res['amends'])
        return res
        

    def getSQLbyAmends(self,evalObj):
        systemPrompt,prompt=self.getPrompt_getSQLbyAmends(evalObj)
        res=self.testClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            evalObj['amend_sql'].append('error')
            return res 
        evalObj['amend_sql'].append(res['sql'])
        return res
    
    #start with getting a wrong sql without inputing database schema.   
    def processWrongSQL(self):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            totalCount+=1
            if originObj['amend_sql']:
                print(f'question{self.evalID} has wrong SQL, skip.')
                continue
            evalObj=copy.deepcopy(originObj)
            print("processing question: ",self.evalID,' ',evalObj['question'])
            self.getWrongSQL(evalObj)
            try:
                self.evaluate_amendRes(evalObj)
            except Exception as e:
                print(e)
            if evalObj['amend_res']==1:
                print(f"{GREEN}Success!{RESET}")
                correctCount+=1
            if evalObj['amend_res']==0:
                print(f"{RED}fail{RESET}")
            print(f'{BLUE}ex: {correctCount/totalCount}{RESET}\n----------------------------\n')
            self.evalRes[self.evalID]=copy.deepcopy(evalObj)
            json.dump(self.evalRes,open(self.dump_path,'w'),indent=4) 
            

    def rule_loop(self):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            try:
                totalCount+=1
                evalObj=copy.deepcopy(originObj)
                if evalObj['rule_res']==1:
                    print(f'question{self.evalID} is able to output right rule, skip.')
                    correctCount+=1
                    continue
                if evalObj['amend_res']==0:
                    print(f'question{self.evalID} require right amends, skip.')
                    continue
                if len(evalObj['rules']):
                    print(f'question{self.evalID} has failed before,skip.')
                    continue
                evalObj['rules']=[]
                print("processing question: ",self.evalID,' ',evalObj['question'])
                self.getRules(evalObj)
                self.getSQLbyRules(evalObj)
                self.evaluate_ruleRes(evalObj)
                if evalObj['rule_res']==1:
                    print(f"{GREEN}Success!{RESET} rules corrected")
                    correctCount+=1
                if evalObj['rule_res']==0:
                    print(f"{RED}fail:{RESET}could not get correct rules, giving up.")
                print(f'{BLUE}ex: {correctCount/totalCount}{RESET}\n----------------------------\n')
                self.evalRes[self.evalID]=copy.deepcopy(evalObj)
                json.dump(self.evalRes,open(self.dump_path,'w'),indent=4) 
            except Exception as e:
                print(e)
                continue

    def amend_loop(self,maxRefineloop=5):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            totalCount+=1
            evalObj=copy.deepcopy(originObj)
            if evalObj['rule_res']==1 or evalObj['amend_res']==1:
                print(f'question{self.evalID} is able to output right result, skip.')
                correctCount+=1
                continue
            if len(evalObj['amends'])>=1:
                print(f'question{self.evalID} has failed before,skip.')
                continue
            evalObj['amends']=[]
            print("processing question: ",self.evalID,' ',evalObj['question'])
            loop_count=0
            try:
                while evalObj['amend_res']==0:
                    print('round',loop_count+1)
                    if len(evalObj['amends'])>1 and evalObj['amends'][-1]==evalObj['amends'][-2]:# if identical Reason was generated, probably fail.
                        print(f"identical amends generated, giving up.")
                        print(f'ex: {BLUE}{correctCount/totalCount}{RESET}\n----------------------------\n')
                        break
                    self.getAmends(evalObj)
                    self.getSQLbyAmends(evalObj)
                    self.evaluate_amendRes(evalObj)
                    loop_count+=1
                    if evalObj['amend_res']==1:
                        correctCount+=1
                        print('ex: ',correctCount/totalCount,'\n----------------------------\n')
                    if evalObj['amend_res']==0 and loop_count>=maxRefineloop:
                        print(f"{RED}fail:{RESET}could not correct sql in {maxRefineloop} loops, giving up.")
                        print(f'ex: {BLUE}{correctCount/totalCount}{RESET}\n----------------------------\n')
                        break
                    if evalObj['amend_res']==-1:
                        print(f"{RED}gold_sql error happened{RESET}")
                        totalCount-=1
                        break
                print(f'ex: {BLUE}{correctCount/totalCount}{RESET}\n----------------------------\n')
                self.evalRes[self.evalID]=copy.deepcopy(evalObj)
                json.dump(self.evalRes,open(self.dump_path,'w'),indent=4)
            except Exception as e:  
                print(e)
                continue            


class MCPClient:
    def __init__(self, server_path: str):
        self.client = OpenAI()
        self.server_path = server_path
        self.available_tools = []
        self._load_tools_once()

    def _load_tools_once(self) -> None:
        asyncio.run(self._load_tools())

    async def _load_tools(self) -> None:

        server_params = StdioServerParameters(
            command="uv",
            args=["run", self.server_path],
            env=None,
        )

        async with AsyncExitStack() as stack:
            stdio, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()
            response = await session.list_tools()

            self.available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    },
                }
            for tool in response.tools
            ]

    def connect_gpt(
        self,
        systemPrompt: str,
        prompt: str,
        engine: str = "deepseek-chat",
        max_tokens: int = 512,
        temperature: float = 0,
        stop=None,
    ):
        result = self.client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": prompt},
            ],
            tools=self.available_tools,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        choice = result.choices[0]
        # print(choice)
        if getattr(choice, "finish_reason", None) == "tool_calls":
            res = {}
            for tc in choice.message.tool_calls:
                res.update(json.loads(tc.function.arguments))
            return res
        warnings.warn('tool did not call.')
        return {"error": "something went wrong, tool did not call."}