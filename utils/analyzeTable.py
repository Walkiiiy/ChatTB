import sqlite3
import json
import math
import os
import random

def safe_decode(val):
    if isinstance(val, bytes):
        for enc in ("utf-8", "latin1", "cp1254"):
            try:
                return val.decode(enc)
            except UnicodeDecodeError:
                continue
        return val.decode("utf-8", errors="replace")
    return val

def safe_float(val):
    try:
        f = float(val)
        if math.isinf(f) or math.isnan(f):
            return None
        return f
    except (ValueError, TypeError, OverflowError):
        return None

def analyze_sqlite_schema(db_path, sample_size=5, sample_rows=1000):
    conn = sqlite3.connect(db_path)
    conn.text_factory = bytes
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [safe_decode(row[0]) for row in cursor.fetchall()]
    result = {}

    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = cursor.fetchall()

        cursor.execute(f'PRAGMA foreign_key_list("{table}")')
        fks = cursor.fetchall()
        fk_map = {safe_decode(fk[3]): {"toTable": safe_decode(fk[2]), "toColumn": safe_decode(fk[4])} for fk in fks}

        # Fetch a sample of rows for this table
        cursor.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}')
        rows = cursor.fetchall()
        columns_names = [safe_decode(col[1]) for col in columns]

        # Transpose rows to column-wise data
        col_values_map = {col: [] for col in columns_names}
        for row in rows:
            for col, val in zip(columns_names, row):
                col_values_map[col].append(safe_decode(val))

        for col in columns:
            col_name = safe_decode(col[1])
            col_type = safe_decode(col[2])
            values = col_values_map[col_name]

            total_count = len(values)
            empty_count = sum(1 for v in values if v is None)

            non_null_values = [v for v in values if v is not None]

            samples = random.sample(non_null_values, min(sample_size, len(non_null_values))) if non_null_values else []

            numeric_values = [safe_float(v) for v in non_null_values if safe_float(v) is not None]

            avg_val = max_val = min_val = var_val = None
            type_num = 2 if numeric_values else 1

            if numeric_values:
                avg_val = sum(numeric_values) / len(numeric_values)
                max_val = max(numeric_values)
                min_val = min(numeric_values)
                var_val = sum((x - avg_val) ** 2 for x in numeric_values) / len(numeric_values) if len(numeric_values) > 1 else 0.0

            unique_count = len(set(non_null_values))
            val_type = "discrete types" if total_count > 0 and unique_count < total_count / 2 else "continuous values"

            def clean_number(x):
                if x is None:
                    return None
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return None
                return x

            result[col_name] = {
                "originColumnName": col_name,
                "belongsToTable": table,
                "dataFormat": col_type,
                "size": total_count,
                "emptyValueCount": empty_count,
                "valType": val_type,
                "typeNum": type_num,
                "samples": samples,
                "averageValue": clean_number(avg_val),
                "maximumValue": clean_number(max_val),
                "minimumValue": clean_number(min_val),
                "sampleVariance": clean_number(var_val),
                "foreignKey": fk_map.get(col_name, None)
            }

    conn.close()
    return result

def find_sqlite_files(parent_dir):
    sqlite_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".sqlite"):
                sqlite_files.append(os.path.join(root, file))
    return sqlite_files

if __name__ == "__main__":
    DATABASE = '/home/walkiiiy/ChatTB/Bird_train/train_databases'
    res = {}
    for database in find_sqlite_files(DATABASE):
        schema = analyze_sqlite_schema(database)
        db_name = database.split('/')[-1].split('.')[0]
        res[db_name] = schema
        print(db_name, 'processed.')

    with open('/home/walkiiiy/ChatTB/Bird_train/train_schema.json', 'w', encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
