from collections import defaultdict
from typing import List, Dict, Any
import os
import json

def evalres(testRange):
    # f=open('/home/walkiiiy/ChatTB/Evaluation/exp_result/V9_output/dev_eval_looped150.json')
    f=open('/home/walkiiiy/ChatTB/Process_document/V14/amends_res.json')
    # f=open('/home/walkiiiy/ChatTB/Evaluation/exp_result/V9_output/test_looped_150.json')
    j=json.load(f)
    succeed=0
    for i  in range(testRange):
        # print(j[en])
        res=j[str(i)]["res"]
        if res==1:
            succeed+=1
    rate=succeed/testRange
    print('total test: ',testRange,'\ntotal ac: ',succeed,'\nex: ',rate)
evalres(1533)
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
# shuffle_and_split_json('/home/walkiiiy/ChatTB/Process_document/V14/dev.json',
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
# merge_and_sort_json_files(
#     [
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_1.json',
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_2.json',
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_3.json',
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_4.json',
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_5.json',
#         '/home/walkiiiy/ChatTB/Process_document/V14/part_6.json'    ],
#         '/home/walkiiiy/ChatTB/Process_document/V14/amends_res.json'
# )


def  clear_reasons():
    f=open('/home/walkiiiy/ChatTB/Process_document/exp_result/v11_output/self_evidence_processed.json')
    j=json.load(f)
    for obj in j:
        j[obj]['reason']=[]
    f=open('/home/walkiiiy/ChatTB/Process_document/exp_result/v11_output/self_evidence_processed.json',"w")
    json.dump(j,f,indent=4)

def delete_reasons():
    # 删除所有的reason
    f=open('/home/walkiiiy/ChatTB/Process_document/V11/dev.json')
    j=json.load(f)
    for obj in j:
        if 'reason' in j[obj]:
            del j[obj]['reason']
    f=open('/home/walkiiiy/ChatTB/Process_document/V11/dev.json',"w")
    json.dump(j,f,indent=4)
    print("All reasons deleted.")


def change_evidence_to_reason():
    # 将所有的evidence改为reason
    f=open('/home/walkiiiy/ChatTB/Process_document/V12/dev.json')
    j=json.load(f)
    for obj in j:
        j[obj]['solution'] = []
    f=open('/home/walkiiiy/ChatTB/Process_document/V12/dev.json',"w")
    json.dump(j,f,indent=4)
# change_evidence_to_reason()
def solution_extract():
    f=open('/home/walkiiiy/ChatTB/Process_document/V14/dev.json')
    j=json.load(f)
    for obj in j:
        j[obj]['sql'] =[j[obj]['sql']]
    f=open('/home/walkiiiy/ChatTB/Process_document/V14/dev.json',"w")
    json.dump(j,f,indent=4)
# solution_extract()