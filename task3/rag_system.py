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
    """
    Initialize the RAG System    
    Args:
        embedding_model_name: Embedding Model Name
        milvus_host: Milvus Local Server Name
        milvus_port: Milvus Local Server Port
        collection_name: Milvus Collection Name
        embeddings_file: Local File Path to the stored embedded file
    Return: N/A
    """
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-large-zh",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 collection_name: str = "city_data",
                 embeddings_file: str = "embeddings_cache/merged_data_embeddings.json"):

        self.embedder = TextEmbedder(model_name=embedding_model_name)
        
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embeddings_file = embeddings_file
        
        connections.connect(host=milvus_host, port=milvus_port)            # Connect to the Local server
        
        # Initialize BM25
        self.bm25 = None
        self.corpus = []
        self.documents = []
        
    def preprocess_text(self, text: str) -> str:
        words = jieba.cut(text)
        filtered_words = [word for word in words if len(word.strip()) > 1]
        return " ".join(filtered_words)
    
    def load_embeddings(self):
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
        
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Loaded {len(self.documents)} documents with embeddings")
    
    def create_milvus_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        schema = {
            "fields": [
                {"name": "id", "dtype": "INT64", "is_primary": True},
                {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": len(self.embeddings[0])},
                {"name": "title", "dtype": "VARCHAR", "max_length": 256},
                {"name": "content", "dtype": "VARCHAR", "max_length": 65535}
            ]
        }
        collection = Collection(name=self.collection_name, schema=schema)            # Create the collection
        
        # Creating the index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        return collection
    
    def index_data(self):
        print("Indexing data to Milvus...")
        collection = self.create_milvus_collection()
        
        batch_size = 1000
        total_docs = len(self.documents)
        
        for i in tqdm(range(0, total_docs, batch_size), desc="Indexing"):
            batch_docs = self.documents[i:i+batch_size]
            batch_embeddings = self.embeddings[i:i+batch_size]
            
            entities = [
                list(range(i, i + len(batch_docs))),    # ids
                batch_embeddings,                       # embeddings
                [doc['title'] for doc in batch_docs],   # titles
                [doc['content'] for doc in batch_docs]  # contents
            ]
            collection.insert(entities)
        
        # Load the collection
        collection.load()
        print("Indexing completed!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        processed_query = self.preprocess_text(query)
        
        tokenized_query = processed_query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
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
        
        # Merge Result
        vector_results = []
        for hits in results:
            for hit in hits:
                vector_results.append({
                    "title": hit.entity.get('title'),
                    "content": hit.entity.get('content'),
                    "score": hit.score
                })
        
        bm25_results = [self.documents[idx] for idx in bm25_indices]        # Merge both result come back
        
        all_results = []
        seen = set()
        
        for result in vector_results + bm25_results:
            key = (result['title'], result['content'])
            if key not in seen:
                seen.add(key)
                all_results.append(result)
        
        return all_results[:top_k]

def main():
    rag = RAGSystem()                # Initialize the RAG system
    rag.load_embeddings()            # Load the stored embedding file
    rag.index_data()                 # Index all data
    
    # Sample Search
    while True:
        query = input("\nPlease input the content you want to search（Input 'quit' to Quit）: ")
        if query.lower() == 'quit':
            break
            
        results = rag.search(query)
        print("\nSearch Result:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"   Content: {result['content'][:200]}...")
            if 'score' in result:
                print(f"    Similarties Score: {result['score']:.4f}")

if __name__ == "__main__":
    main() 
