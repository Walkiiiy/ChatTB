import os
import json
import logging
from typing import Optional, Dict, List, Any
import re
import sqlite3

logger = logging.getLogger(__name__)
class SchemaInformation:
    def __init__(self, table_schema_path=None):
        if table_schema_path:
            with open(table_schema_path, 'r', encoding='utf-8') as f:
                self.table_schema = json.load(f)
        else:
            self.table_schema = {}

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
        unfound_info=""
        # print(self.table_schema)
        for word in words:
            if word in self.table_schema[db_name]:
                schemas.append(self.table_schema[db_name][word])
            else:
                logger.warning(f"Word in condition: {word} not found in database {db_name}")
                unfound_info+=f'''Word in condition: "{word}" not found in database {db_name},
                \if it's supposed to be a column or table name,
                the condition should be considered to revise with correct name in schema.\n
                '''

        return self.schema_to_natural_language(schemas),unfound_info
        
    def generate_specific_column_info(self,db_path, column_names:list):
        db_name = os.path.basename(db_path).split('.')[0]
        schemas=[]
        unfound_info={}
        for column in column_names:
            if column in self.table_schema[db_name]:
                schemas.append(self.table_schema[db_name][column])
            else:
                logger.warning(f"Column in condition: {column} not found in database {db_name}")
                unfound_info[column]=f'''Column in condition: "{column}" not found in database {db_name},
                \if it's supposed to be a column name,
                the condition should be considered to revise with correct name in schema.\n
                '''
        res=self.schema_to_natural_language(schemas)
        res.update(unfound_info)
        return res

        
    def generate_schema_info(self,db_path, num_rows=None):
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
    
    def extract_constraints(self, db_path: str) -> Dict[str, Any]:
        """
        Extract primary keys, foreign keys, and constraints for each table.
        Returns dict {table_name: {"columns": [...], "primary_keys": [...], "foreign_keys": [...]}}
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        schema_info = {}

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']

        for table in tables:
            schema_info[table] = {"columns": [], "primary_keys": [], "foreign_keys": []}

            # columns
            cursor.execute(f"PRAGMA table_info({table})")
            for col in cursor.fetchall():
                cid, name, col_type, notnull, dflt_value, pk = col
                schema_info[table]["columns"].append({"name": name, "type": col_type})
                if pk:
                    schema_info[table]["primary_keys"].append(name)

            # foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            for fk in cursor.fetchall():
                id_, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                schema_info[table]["foreign_keys"].append({
                    "from": from_col,
                    "to_table": ref_table,
                    "to_column": to_col
                })

        conn.close()
        return schema_info

    def schema_to_prompt(self, db_path: str, num_rows: Optional[int] = 5) -> str:
        """
        Convert schema into Arctic-Text2SQL-R1 style serialization:
        - CREATE TABLE statement
        - Columns with PK/FK annotations
        - Optional few-shot rows
        """
        constraints = self.extract_constraints(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        prompt_parts = []

        for table, info in constraints.items():
            # base CREATE TABLE
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
            create_stmt = cursor.fetchone()[0]

            # enrich with constraints
            col_descs = []
            for col in info["columns"]:
                line = f"- {col['name']} ({col['type']})"
                if col['name'] in info["primary_keys"]:
                    line += " [PRIMARY KEY]"
                fk = next((fk for fk in info["foreign_keys"] if fk["from"] == col["name"]), None)
                if fk:
                    line += f" [FK -> {fk['to_table']}({fk['to_column']})]"
                col_descs.append(line)

            desc = f"Table {table}:\n" + "\n".join(col_descs)

            # add sample rows if requested
            rows_prompt = ""
            if num_rows:
                cursor.execute(f"SELECT * FROM {table} LIMIT {num_rows}")
                col_names = [d[0] for d in cursor.description]
                values = cursor.fetchall()
                rows_prompt = self.nice_look_table(col_names, values)
                rows_prompt = f"\n/* {num_rows} sample rows:\n{rows_prompt}\n*/"

            prompt_parts.append(f"{create_stmt}\n{desc}{rows_prompt}")

        conn.close()
        return "\n\n".join(prompt_parts)

