from collections import defaultdict
from typing import List, Dict, Any
import os
import json

AllQueriesPath="/home/walkiiiy/ChatTB/Bird_dev/dev.json"
QueriesPath='/home/walkiiiy/ChatTB/Bird_dev/dev_queries'

def split_questions_by_db_id(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Splits input data by 'db_id', keeping only 'question' and 'evidence' fields.

    Args:
        data (List[Dict[str, Any]]): List of JSON-like dictionaries with at least 'db_id', 'question', and 'evidence'.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary where each key is a db_id and each value is a list of
                                         dicts with only 'question' and 'evidence'.
    """
    db_map = defaultdict(list)
    for entry in data:
        # if entry['difficulty']=='simple':
        #     continue
        db_id = entry.get("db_id")
        if db_id and "question" in entry and "evidence" in entry:
            db_map[db_id].append({
                "question": entry["question"],
                "evidence": entry["evidence"],
                "difficulty":entry['difficulty']
            })
    return dict(db_map)

def split_queries():
    with open(AllQueriesPath) as f:
        all_queries=json.load(f)
    res=split_questions_by_db_id(all_queries)
    for key in res:
        with open(os.path.join(QueriesPath,key+'.json'),'w')as f:
            json.dump(res[key],f,indent=4)

def fetch_prompts():
    res=[]
    breifDescLis=[]
    with open('/home/walkiiiy/ChatTB/prompt_fetchCleanDocument.md')as f:
        prompt=f.read()
    for descfile in os.listdir('/home/walkiiiy/ChatTB/Bird_dev/dev_documentsNLdesc'):
        with open(os.path.join('/home/walkiiiy/ChatTB/Bird_dev/dev_documentsNLdesc',descfile))as f:
            fulldesc=f.read()
        breifDescLis.append(fulldesc.split('\n')[0])
    print(breifDescLis)
        
split_queries()