import json
import numpy as np
from embedding import TextEmbedder
from tqdm import tqdm
import os
import heapq

def load_embeddings(file_path: str = "embeddings_cache/merged_data_embeddings.json"):
    print(f"Loading embeddings from {file_path}...")
    file_size = os.path.getsize(file_path) / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} documents")
        return data

def search(query: str, embeddings_data: list, top_k: int = 5):
    embedder = TextEmbedder()
    
    print("Generating query embedding...")
    query_embedding = embedder.encode(query)
    
    top_results = []
    
    print("Processing documents...")
    total_chunks = sum(len(doc['chunks']) for doc in embeddings_data)
    pbar = tqdm(total=total_chunks, desc="Processing chunks")
    
    for doc_idx, doc in enumerate(embeddings_data):
        for chunk in doc['chunks']:
            # Calculate the similarties for the current chunk
            chunk_embedding = np.array(chunk['embedding'])
            similarity = float(np.dot(chunk_embedding, query_embedding.T).flatten()[0])
            
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
    
    # Return the sorted result
    return [item[1] for item in sorted(top_results, reverse=True)]

def main():
    embeddings_data = load_embeddings()
    
    # Test Qyery that related to Travel
    test_queries = [
        # Chinese
        "旅游景点推荐",
        "热门旅游城市",
        "旅游攻略",
        "最佳旅游季节",
        # English
        "tourist attractions",
        "popular travel destinations",
        "travel guide",
        "best time to travel",
        # Mixed Language Query
        "旅游景点 tourist spots",
        "travel 攻略"
    ]
    
    # Process the search
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(query, embeddings_data)
        
        # Print the result
        print("\nTop results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"Language: {result['language']}")
            print("---")

if __name__ == "__main__":
    main() 
