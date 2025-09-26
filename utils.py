from collections import defaultdict
from typing import List, Dict, Any
import os
import json

def evalres(res_path,testRange=0):
    # f=open('/home/walkiiiy/ChatTB/Evaluation/exp_result/V9_output/dev_eval_looped150.json')
    f=open(res_path)
    # f=open('/home/walkiiiy/ChatTB/Evaluation/exp_result/V9_output/test_looped_150.json')
    j=json.load(f)
    succeed=0
    if not testRange:
        testRange=len(j)
    for i  in range(testRange):
        # print(j[en])
        res=j[str(i)]["rule_res"]
        if res==1:
            succeed+=1
    rate=succeed/testRange
    print('total test: ',testRange,'\ntotal ac: ',succeed,'\nex: ',rate)
# evalres('/home/walkiiiy/ChatTB/Spider_train/res.json')
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
    chunk_size = total // 6
    remainder = total % 6

    start = 0
    for i in range(6):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_items = items[start:end]

        # 再转回 dict
        chunk_dict = {k: v for k, v in chunk_items}

        output_file = os.path.join(output_dir, f"part_{i+1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_dict, f, ensure_ascii=False, indent=2)

        print(f"已生成: {output_file} ({len(chunk_dict)} 条记录)")
        start = end
# shuffle_and_split_json('/home/walkiiiy/ChatTB/Process_document/V14/rules_res.json',
#                        '/home/walkiiiy/ChatTB/Process_document/V14/'
#                        )
import json
import os
from typing import List, Union

def merge_and_sort_json_files(
    input_files: Union[str, List[str]],
    output_file: str,
    encoding: str = "utf-8",
    strict_numeric_keys: bool = True,
) -> None:
    """
    将指定结构（顶层为 {str_number: object}）的 JSON 文件合并 -> 按 key 数值排序 -> 写入新的 JSON 文件。
    为避免跨文件的键冲突，输出会把所有项按排序结果重新编号为 "0".."N-1"。

    参数：
        input_files: 单个文件路径，或文件路径列表。
        output_file: 输出 JSON 文件路径（若上级目录不存在会自动创建）。
        encoding: 文件编码。
        strict_numeric_keys: 若为 True，遇到不能转成整数的 key 将报错；False 时将跳过这些 key。

    结果：
        在 output_file 写出一个 dict，键为 "0".."N-1" 的字符串，值为原始对象。
    """
    # 统一为列表
    if isinstance(input_files, str):
        input_files = [input_files]

    # 收集 (int_key, value)
    collected = []
    for path in input_files:
        with open(path, "r", encoding=encoding) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{path} 的顶层必须是 dict（形如 {{'0': {{...}}, '1': {{...}}}}）")

        for k, v in data.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                if strict_numeric_keys:
                    raise ValueError(f"{path} 中的键 {k!r} 不是数值字符串，无法排序。")
                else:
                    # 非数值键选择跳过
                    continue
            collected.append((ik, v))

    # 按数值键排序
    collected.sort(key=lambda t: t[0])

    # 重新编号为 "0".."N-1"
    merged_renumbered = {str(i): v for i, (_, v) in enumerate(collected)}

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # 写出
    with open(output_file, "w", encoding=encoding) as f:
        json.dump(merged_renumbered, f, ensure_ascii=False, indent=2)
    evalres(output_file)

# merge_and_sort_json_files(
#     [
#         '/home/walkiiiy/ChatTB/Bird_train/part_1.json',
#         '/home/walkiiiy/ChatTB/Bird_train/part_2.json',
#         '/home/walkiiiy/ChatTB/Bird_train/part_3.json',
#         '/home/walkiiiy/ChatTB/Bird_train/part_4.json',  ],
#         '/home/walkiiiy/ChatTB/Bird_train/res.json'
# )

def merge_res_json(input_files, output_file):
    merged = {}
    id=0
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            data[item]['id']=item
            data[item]['origin_dataset']=path.split('/')[-2]
            merged.update({str(id):data[item]})
            id+=1
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    evalres(output_file)
# merge_res_json(
#     ['/home/walkiiiy/ChatTB/Bird_dev/res.json',
#     '/home/walkiiiy/ChatTB/Bird_train/res.json',
#     '/home/walkiiiy/ChatTB/Spider_dev/res.json',
#     '/home/walkiiiy/ChatTB/Spider_train/res.json',],
#     '/home/walkiiiy/ChatTB/rules.json'
# )
def extract_rules(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules_dict = {}
    for item in data:
        db_id = data[item].get('db_id')
        rules = data[item].get('rules', [])
        if db_id not in rules_dict:
            rules_dict[db_id] = []
        rules_dict[db_id]+=rules
    # Convert sets to lists for JSON serialization
    # rules_dict = {k: list(v) for k, v in rules_dict.items()}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rules_dict, f, ensure_ascii=False, indent=2)
# extract_rules(
#     '/home/walkiiiy/ChatTB/rules.json',
#     '/home/walkiiiy/ChatTB/rules_extracted.json'
# )
def prepare_json(json_path):
    f=open(json_path)
    j=json.load(f)
    k={}
    for id,obj in enumerate(j):
        
        obj['ground_truth']=obj.pop('SQL')
        obj['amends']=[]
        obj['rules']=[obj.pop('evidence')] if obj['evidence'] else []
        obj['amends_res']=0
        obj['rule_res']=0
        obj['amend_sql']=[]
        obj['rule_sql']=[]
        # obj.pop('difficulty')
        # obj.pop('question_id')
        
        k[str(id)]=obj

    f=open(json_path,"w")
    json.dump(k,f,indent=4)

    
def tweakStructure(json_path):
    f=open(json_path)
    j=json.load(f)
    k={}
    for id,obj in j.items():
        obj['rule_res']=0
        obj['rule_sql']=[]
        if obj['rules']:
            if obj['rules'][0][:2]=='1)':
                obj['rules']=[]
            else:
                obj['rules']=obj['rules'][:1] # 只保留第一条rule

        # obj.pop('amends_res')
        # obj.pop('difficulty')
        # obj.pop('question_id')
        
        k[id]=obj

    f=open(json_path,"w")
    json.dump(k,f,indent=4)

# tweakStructure('Spider_dev/dev_res.json')






import re
def extractRules(inputPath,ouputPath):
    f=open(inputPath)
    j=json.load(f)# 用正则匹配: 一个或多个数字 + 括号
    for id,obj in j.items():
        if obj['rules']:
            p=[]
            for item in obj['rules']:
                item=' '+item
                parts = re.split(r'\s\d+\)\s', item)
                # 去掉空字符串（因为开头会多一个空）
                parts = [p.strip() for p in parts if p.strip()]
                temp=''
                for r in parts[:-1]:
                    temp+=r.split(':')[0]+', '
                parts[-1]=temp+parts[-1] #给ouputs columns加上条件
                print(parts[-1])
                p+=parts
            j[id]['rules']=p
    f=open(ouputPath,'w')
    json.dump(j,f,indent=4)

extractRules(
    'Bird_dev/res.json',
    'Bird_dev/condensed_rules.json'
)



def structure_rules(inputPath,outputPath):
    f=open(inputPath)
    j=json.load(f)
    k={}
    
    for schema in j:
        id=0
        k[schema]={}
        for rule in j[schema]:
            splited=rule.split(':',1)
            if len(splited)>2:
                print(splited)
            condition=splited[0]
            operation=splited[-1]
            k[schema][str(id)]={}
            k[schema][str(id)]['condition']=condition
            k[schema][str(id)]['operation']=operation
            id+=1
    f=open(outputPath,'w')
    json.dump(k,f,indent=4)

# structure_rules(
#     'Spider_train/rules.json',
#     'Spider_train/rules.json'
# )

import sqlite3
def prepare_trainSet():
    f=open('/home/walkiiiy/ChatTB/Bird_train/train_cleaned.json')
    j=json.load(f)
    i=0
    toBePop=[]
    for item in j:
        if "evidence" in j[item]:
            j[item].pop('evidence')
        j[item]['rules']=[]
    f=open('/home/walkiiiy/ChatTB/Bird_train/train_cleaned.json','w')
    json.dump(j,f,indent=4)
# prepare_trainSet()

def chunkTopSimilarity():
    with open('/home/walkiiiy/ChatTB/Bird_train/rules_conditionSimilarity.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for db in data:
        for item in data[db]:
            del_lis=[]
            for similar_id in data[db][item]['similar_rules']:
                score = data[db][item]['similar_rules'][similar_id]
                if score < 0.7:
                    del_lis.append(similar_id)
            for del_id in del_lis:
                del data[db][item]['similar_rules'][del_id]
    with open('/home/walkiiiy/ChatTB/Bird_train/rules_conditionSimilarity>0.7.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# chunkTopSimilarity()

def tweak_similarRule_json():
    f=open('/home/walkiiiy/ChatTB/Bird_train/rules_conditionSimilarity>0.7.json')
    j=json.load(f)
    for db in j:
        for rid in j[db]:
            j[db][rid]['similar_condition']=j[db][rid].pop('similar_rules')
    f=open('/home/walkiiiy/ChatTB/Bird_train/rules_conditionSimilarity>0.7.json','w')
    json.dump(j,f,indent=4)



def compare_sql_with_db_id(sql1: str, sql2: str, db_id: str, db_root_path: str) -> dict:
    """
    Compare if two SQL queries produce the same results using database ID.
    
    Args:
        sql1: First SQL query to compare
        sql2: Second SQL query to compare
        db_id: Database identifier (used to construct db_path)
        db_root_path: Root path to databases directory
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        from Process_model.SQLTestComparator import SQLTestComparator
        
        comparator = SQLTestComparator(db_root_path)
        
        # Use the test_sql_with_db_id method
        result_code = comparator.test_sql_with_db_id(sql1, sql2, db_id)
        
        # Construct full db_path for detailed comparison
        db_path = os.path.join(db_root_path, db_id, db_id + '.sqlite')
        detailed_info = comparator.compare_results_detailed(sql1, sql2, db_path)
        
        return {
            "match": result_code == 1,
            "result_code": result_code,
            "error": detailed_info.get("error"),
            "detailed_info": detailed_info,
            "db_id": db_id,
            "db_path": db_path
        }
        
    except Exception as e:
        return {
            "match": False,
            "result_code": -1,
            "error": f"Error during comparison: {str(e)}",
            "detailed_info": {},
            "db_id": db_id,
            "db_path": None
        }




# print(compare_sql_with_db_id(
#     '''SELECT T2.Phone
# FROM frpm AS T1
# INNER JOIN schools AS T2
#     ON T1.CDSCode = T2.CDSCode
# WHERE T1."Charter School (Y/N)" = 1
#   AND T1."Charter Funding Type" = 'Directly funded'
#   AND T2.OpenDate > '2000-01-01';
# ''',
# '''
# SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`Charter School (Y/N)` = 1 AND T2.OpenDate > '2000-01-01'
# ''',
# 'california_schools',
# '/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
# ))