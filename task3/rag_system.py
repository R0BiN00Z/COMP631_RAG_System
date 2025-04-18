import json
import jieba
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
from pymilvus import connections, Collection, utility
from embedding import TextEmbedder
import os

class RAGSystem:
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-large-zh",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 collection_name: str = "city_data",
                 embeddings_file: str = "embeddings_cache/merged_data_embeddings.json"):
        """
        初始化 RAG 系统
        
        Args:
            embedding_model_name: 嵌入模型名称
            milvus_host: Milvus 服务器主机
            milvus_port: Milvus 服务器端口
            collection_name: Milvus 集合名称
            embeddings_file: 预生成的嵌入向量文件路径
        """
        # 初始化文本嵌入器（仅用于查询）
        self.embedder = TextEmbedder(model_name=embedding_model_name)
        
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embeddings_file = embeddings_file
        
        # 连接 Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        
        # 初始化 BM25
        self.bm25 = None
        self.corpus = []
        self.documents = []
        
    def preprocess_text(self, text: str) -> str:
        """预处理文本，包括分词和清理"""
        # 分词
        words = jieba.cut(text)
        # 过滤停用词和特殊字符
        filtered_words = [word for word in words if len(word.strip()) > 1]
        return " ".join(filtered_words)
    
    def load_embeddings(self):
        """加载预生成的嵌入向量"""
        print(f"Loading embeddings from {self.embeddings_file}...")
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = []
        self.corpus = []
        self.embeddings = []
        
        for item in tqdm(data, desc="Processing embeddings"):
            self.documents.append({
                'title': item['title'],
                'content': item['content']
            })
            self.corpus.append(f"{item['title']} {item['content']}")
            self.embeddings.append(item['embedding'])
        
        # 初始化 BM25
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Loaded {len(self.documents)} documents with embeddings")
    
    def create_milvus_collection(self):
        """创建 Milvus 集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        # 定义集合模式
        schema = {
            "fields": [
                {"name": "id", "dtype": "INT64", "is_primary": True},
                {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": len(self.embeddings[0])},
                {"name": "title", "dtype": "VARCHAR", "max_length": 256},
                {"name": "content", "dtype": "VARCHAR", "max_length": 65535}
            ]
        }
        
        # 创建集合
        collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        return collection
    
    def index_data(self):
        """将数据索引到 Milvus"""
        print("Indexing data to Milvus...")
        collection = self.create_milvus_collection()
        
        # 准备批量插入的数据
        batch_size = 1000
        total_docs = len(self.documents)
        
        for i in tqdm(range(0, total_docs, batch_size), desc="Indexing"):
            batch_docs = self.documents[i:i+batch_size]
            batch_embeddings = self.embeddings[i:i+batch_size]
            
            # 准备插入数据
            entities = [
                list(range(i, i + len(batch_docs))),  # ids
                batch_embeddings,  # embeddings
                [doc['title'] for doc in batch_docs],  # titles
                [doc['content'] for doc in batch_docs]  # contents
            ]
            
            # 插入数据
            collection.insert(entities)
        
        # 加载集合
        collection.load()
        print("Indexing completed!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        # 预处理查询
        processed_query = self.preprocess_text(query)
        
        # BM25 搜索
        tokenized_query = processed_query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # 向量搜索
        query_embedding = self.embedder.encode(query)
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content"]
        )
        
        # 合并结果
        vector_results = []
        for hits in results:
            for hit in hits:
                vector_results.append({
                    "title": hit.entity.get('title'),
                    "content": hit.entity.get('content'),
                    "score": hit.score
                })
        
        # 合并 BM25 和向量搜索结果
        bm25_results = [self.documents[idx] for idx in bm25_indices]
        
        # 去重并排序
        all_results = []
        seen = set()
        
        for result in vector_results + bm25_results:
            key = (result['title'], result['content'])
            if key not in seen:
                seen.add(key)
                all_results.append(result)
        
        return all_results[:top_k]

def main():
    # 初始化 RAG 系统
    rag = RAGSystem()
    
    # 加载预生成的嵌入向量
    rag.load_embeddings()
    
    # 索引数据
    rag.index_data()
    
    # 示例搜索
    while True:
        query = input("\n请输入搜索查询（输入 'quit' 退出）: ")
        if query.lower() == 'quit':
            break
            
        results = rag.search(query)
        print("\n搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 标题: {result['title']}")
            print(f"   内容: {result['content'][:200]}...")
            if 'score' in result:
                print(f"   相似度得分: {result['score']:.4f}")

if __name__ == "__main__":
    main() 