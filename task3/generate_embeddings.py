import json
from embedding import TextEmbedder
from tqdm import tqdm
import os
import numpy as np
import shutil
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import time
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def init_pinecone(api_key: str = None):
    """初始化Pinecone客户端"""
    if api_key is None:
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("Pinecone API key not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    
    # 创建或获取索引
    index_name = os.getenv('PINECONE_INDEX_NAME', 'irnew')
    dimension = 1024  # BAAI/bge-m3模型的维度
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
        print(f"Created new index: {index_name}")
    
    return pc.Index(index_name)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_with_retry(index, vectors):
    """带重试机制的上传函数"""
    return index.upsert(vectors=vectors)

def process_merged_data(input_file: str = "merged_data.json",
                       batch_size: int = 128,
                       chunk_size: int = 300,
                       chunk_overlap: int = 50,
                       title_weight: float = 3.0,
                       pinecone_api_key: str = None,
                       start_from: int = 2846):  # 添加起始位置参数
    """
    处理 merged_data.json 文件，生成嵌入向量并上传到Pinecone
    """
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is required")
    
    # 检查GPU可用性
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    # 初始化Pinecone
    print("Initializing Pinecone...")
    index = init_pinecone(pinecone_api_key)
    
    # 初始化嵌入器，使用GPU
    embedder = TextEmbedder(
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )
    
    # 读取输入数据
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建内容映射文件
    content_mapping = {}
    
    # 准备数据
    print("Preparing data...")
    total_docs = len(data)
    processed_docs = 0
    total_chunks = 0
    
    # 批处理上传
    pinecone_batch_size = 100  # Pinecone的批处理大小
    vectors_batch = []
    
    # 从指定位置开始处理
    for doc_idx in tqdm(range(start_from, total_docs), desc="Processing documents"):
        item = data[doc_idx]
        if item.get('title') and item.get('content'):
            title = item['title']
            content = item['content']
            doc_id = f"doc_{doc_idx}"
            
            # 存储完整内容到映射文件
            content_mapping[doc_id] = {
                'title': title,
                'content': content
            }
            
            # 对内容进行分块
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                chunks.append(chunk)
                start = end - chunk_overlap  # 考虑重叠
            
            # 为每个chunk生成向量
            for chunk_idx, chunk in enumerate(chunks):
                # 准备带权重的标题
                weighted_title = ' '.join([title] * int(title_weight))
                text = f"{weighted_title} {chunk}"
                
                # 生成嵌入向量
                embedding = embedder.encode(text)
                
                # 准备元数据
                metadata = {
                    'title': title,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'language': embedder.detect_language(text)
                }
                
                # 添加到批次
                vectors_batch.append({
                    'id': f"{doc_id}_chunk_{chunk_idx}",
                    'values': embedding.flatten().tolist(),
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                # 当批次达到指定大小时上传
                if len(vectors_batch) >= pinecone_batch_size:
                    print(f"Uploading batch of {len(vectors_batch)} vectors...")
                    try:
                        upsert_with_retry(index, vectors_batch)
                        vectors_batch = []
                        time.sleep(1)  # 添加延迟以避免速率限制
                    except Exception as e:
                        print(f"Error uploading batch: {e}")
                        # 如果上传失败，保存当前进度
                        with open('content_mapping.json', 'w', encoding='utf-8') as f:
                            json.dump(content_mapping, f, ensure_ascii=False, indent=2)
                        raise e
            
            processed_docs += 1
    
    # 上传剩余的向量
    if vectors_batch:
        print(f"Uploading final batch of {len(vectors_batch)} vectors...")
        upsert_with_retry(index, vectors_batch)
    
    # 保存内容映射文件
    with open('content_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(content_mapping, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing complete!")
    print(f"Total documents processed: {processed_docs}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per document: {total_chunks/processed_docs:.2f}")
    print(f"Content mapping saved to content_mapping.json")
    
    # 打印索引统计信息
    index_stats = index.describe_index_stats()
    print("\nPinecone index statistics:")
    print(f"Total vectors: {index_stats['total_vector_count']}")
    print(f"Dimension: {index_stats['dimension']}")
    print(f"Index fullness: {index_stats['index_fullness']}")

def main():
    # 处理数据
    process_merged_data(
        batch_size=128,
        chunk_size=300,
        chunk_overlap=50,
        title_weight=3.0,
        start_from=2846  # 从第2846个文档开始
    )

if __name__ == "__main__":
    main() 