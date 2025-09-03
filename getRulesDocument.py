from Process_document.rule_loop import RuleProcesser
from concurrent.futures import ProcessPoolExecutor

import json
import random
import os
def shuffle_and_split_json(input_file, output_dir):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是 dict 格式")

    # 转换为 (key, value) 列表
    items = list(data.items())

    # 随机打乱
    random.shuffle(items)

    # 计算每个文件的大小
    total = len(items)
    chunk_size = total // 4
    remainder = total % 4

    start = 0
    for i in range(4):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_items = items[start:end]

        # 再转回 dict
        chunk_dict = {k: v for k, v in chunk_items}

        output_file = os.path.join(output_dir, f"part_{i+1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_dict, f, ensure_ascii=False, indent=2)

        print(f"已生成: {output_file} ({len(chunk_dict)} 条记录)")
        start = end

jsonpath='/home/walkiiiy/ChatTB/Spider_train/train.json'
dir='/home/walkiiiy/ChatTB/Spider_train'


# jsonpath='/home/walkiiiy/ChatTB/Spider_dev/dev_res.json'
# dir='/home/walkiiiy/ChatTB/Spider_dev'

# jsonpath='/home/walkiiiy/ChatTB/Bird_dev/dev_wSQLres.json'
# dir='/home/walkiiiy/ChatTB/Bird_dev'

# jsonpath='/home/walkiiiy/ChatTB/Bird_train/train.json'
# dir='/home/walkiiiy/ChatTB/Bird_train'

# shuffle_and_split_json(jsonpath,dir)


def run_process(split):
    splitpath = dir + f'/part_{split}.json'   # 每个split一个文件
    load_path = splitpath
    dump_path = splitpath

    db_root_path = '/home/walkiiiy/ChatTB/Spider_train/database'
    tableSchema_path = '/home/walkiiiy/ChatTB/Spider_train/train_schema.json'

    # db_root_path = '/home/walkiiiy/ChatTB/Spider_dev/database'
    # tableSchema_path = '/home/walkiiiy/ChatTB/Spider_dev/dev_schema.json'

    # db_root_path = '/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
    # tableSchema_path = '/home/walkiiiy/ChatTB/Bird_dev/dev_schema.json'

    # db_root_path = '/home/walkiiiy/ChatTB/Bird_train/train_databases'
    # tableSchema_path = '/home/walkiiiy/ChatTB/Bird_train/train_schema.json'

    Processer = RuleProcesser(
        load_path=load_path,
        dump_path=dump_path,
        db_root_path=db_root_path,
        tableSchema_path=tableSchema_path,
    )
    Processer.processWrongSQL()
    Processer.amend_loop()
    Processer.rule_loop()

if __name__ == "__main__":
    splits = [1,2,3,4]  # 你有几个split就写几个
    with ProcessPoolExecutor(max_workers=len(splits)) as executor:
        executor.map(run_process, splits)