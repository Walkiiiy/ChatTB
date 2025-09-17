import os
import json
import logging
from typing import Optional, Dict, List, Any
import re
import sqlite3

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
        for word in words:
            if word in self.tableSchema[db_name]:
                schemas.append(self.tableSchema[db_name][word])
        return self.schema_to_natural_language(schemas)
    
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
