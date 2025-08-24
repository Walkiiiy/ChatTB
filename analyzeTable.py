import sqlite3
import json
import statistics
import os
import math

def analyze_sqlite_schema(db_path, sample_size=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    result = {}

    for table in tables:
        # 表的列信息
        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

        # 表的外键信息
        cursor.execute(f'PRAGMA foreign_key_list("{table}")')
        fks = cursor.fetchall()  # (id, seq, table, from, to, on_update, on_delete, match)
        fk_map = {fk[3]: {"toTable": fk[2], "toColumn": fk[4]} for fk in fks}

        for col in columns:
            col_name = col[1]
            col_type = col[2]

            cursor.execute(f'SELECT "{col_name}" FROM "{table}"')
            values = [row[0] for row in cursor.fetchall()]

            total_count = len(values)
            empty_count = sum(1 for v in values if v is None)

            non_null_values = [v for v in values if v is not None]

            samples = non_null_values[:sample_size] if non_null_values else []

            # 统计数字型数据
            numeric_values = []
            for v in non_null_values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass

            avg_val = max_val = min_val = var_val = None
            type_num = 0
            if numeric_values:
                type_num = 2
                avg_val = statistics.mean(numeric_values)
                max_val = max(numeric_values)
                min_val = min(numeric_values)
                var_val = statistics.pvariance(numeric_values) if len(numeric_values) > 1 else 0.0
            else:
                type_num = 1

            # 判断 valType
            unique_count = len(set(non_null_values))
            if total_count > 0 and unique_count < total_count / 2:
                val_type = "discrete types"
            else:
                val_type = "continuous values"

            # ⚡ 处理 NaN / Inf -> None
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
                # ⚡ 新增：外键信息
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
    DATABASE = '/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
    res = {}
    for database in find_sqlite_files(DATABASE):
        schema = analyze_sqlite_schema(database)
        db_name = database.split('/')[-1].split('.')[0]
        res[db_name] = schema
        print(db_name, 'processed.')

    with open('/home/walkiiiy/ChatTB/Bird_dev/dev_tableAnalyze.json', 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
