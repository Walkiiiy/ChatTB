from Process_model.SchemaInformation import SchemaInformation
import os
import json

def get_schema_info(db_path, num_rows=None):
    schema_info = SchemaInformation()
    return schema_info.generate_schema_info(db_path, num_rows)

if __name__ == "__main__":
    db_root_path = "/home/ubuntu/walkiiiy/ChatTB/Database_train"
    output_path = "/home/ubuntu/walkiiiy/ChatTB/Database_train/train_schema.json"
    res = {}
    for db_path in os.listdir(db_root_path):
        if not os.path.exists(os.path.join(db_root_path, db_path, f"{db_path}.sqlite")):
            continue
        print('processing', db_path)
        schema=get_schema_info(os.path.join(db_root_path, db_path, f"{db_path}.sqlite"))
        res[db_path] = schema

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print('done')
