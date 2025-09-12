import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class DBRuleSimilarity:
    def __init__(self, 
                 model_name='/home/walkiiiy/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1'):
        # 初始化 Chroma 客户端
        self.client = chromadb.Client()
        # 初始化 embedding 模型
        self.model = SentenceTransformer(model_name)

    def add_database_rules(self, db_name, rules_dict):
        """
        将数据库规则添加到 Chroma 集合
        db_name: 数据库名称 (str)
        rules_dict: { "0": {"condition": "...", "operation": "..."}, ... }
        """
        # 如果已存在集合，删除并重建（确保不会重复添加）
        try:
            self.client.delete_collection(db_name)
        except Exception:
            pass
        
        collection = self.client.create_collection(
            name=db_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

        ids, documents, metadatas = [], [], []
        for rid, item in rules_dict.items():
            # print(rid,item)
            ids.append(str(rid))
            # documents.append(item["condition"])
            documents.append(item["operation"])
            metadatas.append(item)

        # 批量存入
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return collection

    def compute_similarity(self, db_name):
        """
        对数据库内的每条规则，返回与其他规则的top相似度
        """
        collection = self.client.get_collection(db_name)

        # 拿到所有规则
        all_data = collection.get()
        ids, conditions = all_data["ids"], all_data["documents"]

        # 计算 embedding
        embeddings = self.model.encode(conditions, convert_to_numpy=True, normalize_embeddings=True)

        results = {}
        for i, rid in enumerate(ids):
            query_emb = embeddings[i].reshape(1, -1)
            scores = np.dot(embeddings, query_emb.T).squeeze()  # 余弦相似度
            ranking = {ids[j]:float(scores[j]) for j in range(len(ids)) if float(scores[j]) > 0.7},
            results[rid] = ranking
        return results

inputPath = '/home/walkiiiy/ChatTB/Bird_train/rules.json'
ouputPath='/home/walkiiiy/ChatTB/Bird_train/rules_operationSimilarity>0.7.json'
with open(inputPath)as f:
    rules = json.load(f)

SimProcessor = DBRuleSimilarity()
id=1
for db in rules:
    print(id)
    id+=1
    # 把规则存进 Chroma
    SimProcessor.add_database_rules(db_name=db, rules_dict=rules[db])
    ranking = SimProcessor.compute_similarity(db)

    for rid, sim_dict in ranking.items():
        # 初始化相似规则字典
        rules[db][rid]['similar_rules'] = sim_dict

    # 如果只是临时计算，可以删掉集合
    SimProcessor.client.delete_collection(db)
    print('writing...')        
    with open(ouputPath, 'w') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    # break