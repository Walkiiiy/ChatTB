import sqlite3
import json
import statistics

def analyze_sqlite_schema(db_path, sample_size=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    result = {}

    for table in tables:
        # 获取表的列信息
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

        for col in columns:
            col_name = col[1]
            col_type = col[2]

            cursor.execute(f"SELECT {col_name} FROM {table}")
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

            result[col_name] = {
                "originColumnName": col_name,
                "columnDescription": col_name,
                "belongs to table": table,
                "dataFormat": col_type,
                "size": total_count,
                "emptyValueCount": empty_count,
                "valType": val_type,
                "typeNum": type_num,
                "samples": samples,
                "averageValue": avg_val,
                "maximumValue": max_val,
                "minimumValue": min_val,
                "sampleVariance": var_val
            }

    conn.close()
    return result


if __name__ == "__main__":
    db_path = "your_database.sqlite"  # 修改为你的数据库文件路径
    stats = analyze_sqlite_schema(db_path)
    print(json.dumps(stats, indent=4, ensure_ascii=False))
