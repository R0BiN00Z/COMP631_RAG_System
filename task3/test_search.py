import json
import numpy as np
from embedding import TextEmbedder
from tqdm import tqdm
import os
import heapq

def load_embeddings(file_path: str = "embeddings_cache/merged_data_embeddings.json"):
    """加载生成的嵌入向量"""
    print(f"Loading embeddings from {file_path}...")
    file_size = os.path.getsize(file_path) / 1024 / 1024  # 转换为MB
    print(f"File size: {file_size:.2f} MB")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} documents")
        return data

def search(query: str, embeddings_data: list, top_k: int = 5):
    """执行搜索"""
    # 初始化嵌入器
    embedder = TextEmbedder()
    
    # 生成查询的嵌入向量
    print("Generating query embedding...")
    query_embedding = embedder.encode(query)
    
    # 使用堆来维护top-k结果
    top_results = []
    
    print("Processing documents...")
    total_chunks = sum(len(doc['chunks']) for doc in embeddings_data)
    pbar = tqdm(total=total_chunks, desc="Processing chunks")
    
    for doc_idx, doc in enumerate(embeddings_data):
        for chunk in doc['chunks']:
            # 计算当前块的相似度
            chunk_embedding = np.array(chunk['embedding'])
            similarity = float(np.dot(chunk_embedding, query_embedding.T).flatten()[0])
            
            # 使用堆维护top-k结果
            if len(top_results) < top_k:
                heapq.heappush(top_results, (similarity, {
                    'title': doc['title'],
                    'content': doc['content'],
                    'language': doc['language']
                }))
            else:
                if similarity > top_results[0][0]:
                    heapq.heappop(top_results)
                    heapq.heappush(top_results, (similarity, {
                        'title': doc['title'],
                        'content': doc['content'],
                        'language': doc['language']
                    }))
            
            pbar.update(1)
    pbar.close()
    
    # 返回排序后的结果
    return [item[1] for item in sorted(top_results, reverse=True)]

def main():
    # 加载嵌入向量
    embeddings_data = load_embeddings()
    
    # 测试查询 - 旅游相关
    test_queries = [
        # 中文查询
        "旅游景点推荐",
        "热门旅游城市",
        "旅游攻略",
        "最佳旅游季节",
        # 英文查询
        "tourist attractions",
        "popular travel destinations",
        "travel guide",
        "best time to travel",
        # 混合查询
        "旅游景点 tourist spots",
        "travel 攻略"
    ]
    
    # 执行搜索
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(query, embeddings_data)
        
        # 打印结果
        print("\nTop results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"Language: {result['language']}")
            print("---")

if __name__ == "__main__":
    main() 