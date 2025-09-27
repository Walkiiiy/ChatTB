"""
语义相似度比较类
用于比较两个字符串的语义相似度，返回相似度值
支持多种相似度计算方法
"""

import logging
import numpy as np
import torch
from typing import Optional, List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re


class SemanticSimilarity:
    """
    语义相似度比较类
    支持多种相似度计算方法，包括基于embedding的方法和传统文本相似度方法
    """
    
    def __init__(self, 
                 model_name: str = "/home/ubuntu/walkiiiy/ChatTB/Process_model/models--all-MiniLM-L6-v2",
                 use_gpu: bool = True,
                 cache_folder: Optional[str] = None):
        """
        初始化语义相似度比较器
        
        Args:
            model_name: 用于生成embedding的模型名称
            use_gpu: 是否使用GPU加速
            cache_folder: 模型缓存文件夹
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_folder = cache_folder
        
        # 初始化embedding模型
        self.embedding_model = None
        self._load_embedding_model()
        
        # 初始化TF-IDF向量化器（用于传统方法）
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_chinese,
            lowercase=False,
            max_features=5000
        )
        self.tfidf_fitted = False
        
    def _load_embedding_model(self) -> None:
        """加载sentence-transformers模型"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            
            # 尝试加载模型，如果失败则尝试其他模型
            models_to_try = [
                self.model_name,
                "paraphrase-multilingual-MiniLM-L12-v2",  # 支持中文的模型
                "all-MiniLM-L6-v2",  # 英文模型
                "distiluse-base-multilingual-cased"  # 多语言模型
            ]
            
            for model in models_to_try:
                try:
                    self.embedding_model = SentenceTransformer(
                        model,
                        cache_folder=self.cache_folder,
                        device=device
                    )
                    self.model_name = model  # 更新实际使用的模型名
                    self.logger.info(f"Embedding model '{model}' loaded successfully on {device}")
                    return
                except Exception as model_error:
                    self.logger.warning(f"Failed to load model '{model}': {model_error}")
                    continue
            
            # 如果所有模型都失败
            self.logger.error("All embedding models failed to load")
            self.embedding_model = None
            
        except Exception as e:
            self.logger.error(f"Failed to load any embedding model: {e}")
            self.embedding_model = None
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词函数"""
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        # 使用jieba分词
        tokens = list(jieba.cut(text))
        return [token.strip() for token in tokens if token.strip()]
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 转换为字符串并去除多余空白
        text = str(text).strip()
        
        # 移除特殊字符但保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()[]{}]', '', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def cosine_similarity_embedding(self, text1: str, text2: str) -> float:
        """
        使用embedding计算余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度值 (0-1之间，1表示完全相同)
        """
        if self.embedding_model is None:
            self.logger.warning("Embedding model not available, falling back to TF-IDF method")
            return self.cosine_similarity_tfidf(text1, text2)
        
        try:
            # 预处理文本
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            
            if not text1 or not text2:
                return 0.0
            
            # 如果文本完全相同，直接返回1
            if text1 == text2:
                return 1.0
            
            # 生成embedding
            embeddings = self.embedding_model.encode([text1, text2], convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = cosine_similarity(embeddings[0].cpu().numpy().reshape(1, -1), 
                                         embeddings[1].cpu().numpy().reshape(1, -1))[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error in embedding similarity calculation: {e}")
            self.logger.info("Falling back to TF-IDF method")
            return self.cosine_similarity_tfidf(text1, text2)
    
    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """
        使用TF-IDF计算余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度值 (0-1之间，1表示完全相同)
        """
        try:
            # 预处理文本
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            
            if not text1 or not text2:
                return 0.0
            
            # 如果文本相同，直接返回1
            if text1 == text2:
                return 1.0
            
            # 使用TF-IDF向量化
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error in TF-IDF similarity calculation: {e}")
            return 0.0
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        计算Jaccard相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度值 (0-1之间，1表示完全相同)
        """
        try:
            # 预处理文本
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            
            if not text1 or not text2:
                return 0.0
            
            # 分词
            tokens1 = set(self._tokenize_chinese(text1))
            tokens2 = set(self._tokenize_chinese(text2))
            
            if not tokens1 and not tokens2:
                return 1.0
            if not tokens1 or not tokens2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return float(intersection / union) if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in Jaccard similarity calculation: {e}")
            return 0.0
    
    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        计算基于编辑距离的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度值 (0-1之间，1表示完全相同)
        """
        try:
            # 预处理文本
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            
            if not text1 and not text2:
                return 1.0
            if not text1 or not text2:
                return 0.0
            
            # 计算编辑距离
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            
            return float(1 - distance / max_len) if max_len > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in Levenshtein similarity calculation: {e}")
            return 0.0
    
    def calculate_similarity(self, 
                           text1: str, 
                           text2: str, 
                           method: str = "embedding") -> Dict[str, Union[float, str]]:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            method: 计算方法 ("embedding", "tfidf", "jaccard", "levenshtein", "all")
            
        Returns:
            包含相似度值和计算方法的字典
        """
        if method not in ["embedding", "tfidf", "jaccard", "levenshtein", "all"]:
            raise ValueError(f"Unsupported method: {method}")
        
        results = {}
        
        if method == "all":
            # 计算所有方法的相似度
            results["embedding"] = self.cosine_similarity_embedding(text1, text2)
            results["tfidf"] = self.cosine_similarity_tfidf(text1, text2)
            results["jaccard"] = self.jaccard_similarity(text1, text2)
            results["levenshtein"] = self.levenshtein_similarity(text1, text2)
            
            # 计算平均相似度
            similarities = [v for v in results.values() if isinstance(v, (int, float))]
            results["average"] = float(np.mean(similarities)) if similarities else 0.0
            
        elif method == "embedding":
            similarity = self.cosine_similarity_embedding(text1, text2)
            results["similarity"] = similarity
            results["method"] = "embedding"
            
        elif method == "tfidf":
            similarity = self.cosine_similarity_tfidf(text1, text2)
            results["similarity"] = similarity
            results["method"] = "tfidf"
            
        elif method == "jaccard":
            similarity = self.jaccard_similarity(text1, text2)
            results["similarity"] = similarity
            results["method"] = "jaccard"
            
        elif method == "levenshtein":
            similarity = self.levenshtein_similarity(text1, text2)
            results["similarity"] = similarity
            results["method"] = "levenshtein"
        
        return results
    
    def batch_similarity(self, 
                        texts1: List[str], 
                        texts2: List[str], 
                        method: str = "embedding") -> List[Dict[str, Union[float, str]]]:
        """
        批量计算文本相似度
        
        Args:
            texts1: 第一个文本列表
            texts2: 第二个文本列表
            method: 计算方法
            
        Returns:
            相似度结果列表
        """
        if len(texts1) != len(texts2):
            raise ValueError("Text lists must have the same length")
        
        results = []
        for text1, text2 in zip(texts1, texts2):
            result = self.calculate_similarity(text1, text2, method)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息"""
        return {
            "embedding_model": self.model_name,
            "embedding_available": str(self.embedding_model is not None),
            "device": "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        }


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建相似度比较器
    similarity = SemanticSimilarity()
    
    # 测试用例
    test_cases = [
        ("今天天气很好", "天气不错"),
        ("机器学习很有趣", "深度学习很神奇"),
        ("北京是中国的首都", "北京是中国首都"),
        ("我喜欢吃苹果", "我喜欢吃香蕉"),
        ("这是一个测试", "这是一个测试")
    ]
    
    print("语义相似度测试:")
    print("=" * 50)
    
    for text1, text2 in test_cases:
        print(f"文本1: {text1}")
        print(f"文本2: {text2}")
        
        # 使用embedding方法
        result = similarity.calculate_similarity(text1, text2, "embedding")
        print(f"Embedding相似度: {result['similarity']:.4f}")
        
        # 使用所有方法
        all_results = similarity.calculate_similarity(text1, text2, "all")
        print(f"TF-IDF相似度: {all_results['tfidf']:.4f}")
        print(f"Jaccard相似度: {all_results['jaccard']:.4f}")
        print(f"编辑距离相似度: {all_results['levenshtein']:.4f}")
        print(f"平均相似度: {all_results['average']:.4f}")
        print("-" * 30)
    
    # 打印模型信息
    print("\n模型信息:")
    print(similarity.get_model_info())
