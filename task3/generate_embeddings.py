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

load_dotenv()                # Load environment variables from .env file

def init_pinecone(api_key: str = None):
    if api_key is None:
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("Pinecone API key not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    
    index_name = os.getenv('PINECONE_INDEX_NAME', 'irnew')
    dimension = 1024
    
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
    return index.upsert(vectors=vectors)

def process_merged_data(input_file: str = "merged_data.json",
                       batch_size: int = 128,
                       chunk_size: int = 300,
                       chunk_overlap: int = 50,
                       title_weight: float = 3.0,
                       pinecone_api_key: str = None,
                       start_from: int = 2846):

    # Process the merged_data.json file, generate embedding and upload to Pinecone
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is required")
    
    # Check available devices
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    index = init_pinecone(pinecone_api_key)
    
    # Initialize the embedder
    embedder = TextEmbedder(
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )
    
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    content_mapping = {}
    
    print("Preparing data...")
    total_docs = len(data)
    processed_docs = 0
    total_chunks = 0
    
    pinecone_batch_size = 100
    vectors_batch = []
    
    for doc_idx in tqdm(range(start_from, total_docs), desc="Processing documents"):
        item = data[doc_idx]
        if item.get('title') and item.get('content'):
            title = item['title']
            content = item['content']
            doc_id = f"doc_{doc_idx}"
            
            # Store full content into the mapped document
            content_mapping[doc_id] = {
                'title': title,
                'content': content
            }
            
            # Split the text
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                chunks.append(chunk)
                start = end - chunk_overlap
            
            # Generate vertex for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Prepare the title with weighted 
                weighted_title = ' '.join([title] * int(title_weight))
                text = f"{weighted_title} {chunk}"
                
                embedding = embedder.encode(text)            # Generate the embedding
                
                # Prepare the metadata
                metadata = {
                    'title': title,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'language': embedder.detect_language(text)
                }
                
                # Adding into the upload batch
                vectors_batch.append({
                    'id': f"{doc_id}_chunk_{chunk_idx}",
                    'values': embedding.flatten().tolist(),
                    'metadata': metadata
                })
                
                total_chunks += 1
                
                if len(vectors_batch) >= pinecone_batch_size:
                    print(f"Uploading batch of {len(vectors_batch)} vectors...")
                    try:
                        upsert_with_retry(index, vectors_batch)
                        vectors_batch = []
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error uploading batch: {e}")
                        # If upload failed, save the current status
                        with open('content_mapping.json', 'w', encoding='utf-8') as f:
                            json.dump(content_mapping, f, ensure_ascii=False, indent=2)
                        raise e
            
            processed_docs += 1
    
    # Upload the remaining vector
    if vectors_batch:
        print(f"Uploading final batch of {len(vectors_batch)} vectors...")
        upsert_with_retry(index, vectors_batch)
    
    # Save the mapping file
    with open('content_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(content_mapping, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing complete!")
    print(f"Total documents processed: {processed_docs}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per document: {total_chunks/processed_docs:.2f}")
    print(f"Content mapping saved to content_mapping.json")
    
    # Print the index analyze information
    index_stats = index.describe_index_stats()
    print("\nPinecone index statistics:")
    print(f"Total vectors: {index_stats['total_vector_count']}")
    print(f"Dimension: {index_stats['dimension']}")
    print(f"Index fullness: {index_stats['index_fullness']}")

def main():
    process_merged_data(
        batch_size=128,
        chunk_size=300,
        chunk_overlap=50,
        title_weight=3.0,
        start_from=2846 )

if __name__ == "__main__":
    main() 
