import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Tuple
import numpy as np
from tqdm import tqdm
import os
import hashlib
import json
from pathlib import Path
import jieba

class TextEmbedder:
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 batch_size: int = 32,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 device: str = None):
        """
        Initialize text embedder
        
        Args:
            model_name: Model name to use
            batch_size: Batch size for processing
            chunk_size: Maximum length of each text chunk
            chunk_overlap: Overlap length between adjacent chunks
            device: Computing device (cuda/mps/cpu)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Loading model {model_name} on {self.device}...")
        try:
            # First load the model without specifying device
            self.model = SentenceTransformer(model_name)
            # Then move it to the appropriate device
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to CPU if there's an error
            self.device = "cpu"
            self.model = SentenceTransformer(model_name).to(self.device)
        
        self.tokenizer = self.model.tokenizer

    """
    Detect the main Language of text
    Args: text: Input Text
    Returns: Language Code ('zh' or 'en')
    """
    def detect_language(self, text: str) -> str:
        # Simple Language Check, If 30% orr more of the text are chinese. then output zh
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars / len(text) > 0.3:
            return 'zh'
        return 'en'
        
    """
    Split one text into multiple text chunks 
    Args: text: Input Text
    Returns: List of text chunks
    """
    def split_into_chunks(self, text: str) -> List[str]:
        # Detect Language
        lang = self.detect_language(text)
        
        if lang == 'zh':
            # Chinese
            words = list(jieba.cut(text))
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > self.chunk_size:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                        # Keep the Overlayed part
                        overlap = int(self.chunk_overlap / 2)
                        current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                        current_length = sum(len(w) for w in current_chunk)
                
                current_chunk.append(word)
                current_length += len(word)
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
        
        else:
            # English: Spliit by space
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > self.chunk_size:  # +1 for space
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Keep the Overlayed part
                        overlap = int(self.chunk_overlap / 2)
                        current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                        current_length = sum(len(w) + 1 for w in current_chunk) - 1
                
                current_chunk.append(word)
                current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks

    """
    Generate the embedded text
    Args:
        text: Input text
        return_chunks: Return the chunk info or not
    Returns:
        If return_chunks==False, return the encode text
        If return_chunks==True, return the encode text and chunk information
    """
    def encode(self, text: str, return_chunks: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[int, int]]]]:
        # Detect Language
        language = self.detect_language(text)
        
        # Split the text into multiple smaller text chunks
        chunks = self.split_into_chunks(text)
        
        # Generate embedding in the list
        embeddings = []
        chunk_info = []
        
        for i, chunk in enumerate(chunks):
            # Tokenize the text
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(encoded)
                embedding = output['sentence_embedding'].cpu().numpy()
            
            embeddings.append(embedding)
            chunk_info.append((0, i))  # Will always be 0 since we are processing single file at one time
        
        # Assemble all embedded text together
        final_embedding = np.mean(embeddings, axis=0)
        
        if return_chunks:
            return final_embedding, chunk_info
        return final_embedding

    """
    Encode the text with metadata
    Args: texts: List if Dictionary that include text and metadata [{"text": "...", "metadata": {...}}, ...]    
    Returns: List of dicrionary that include embedded data and metadata
    """
    def encode_with_metadata(self, texts: List[dict]) -> List[dict]:
        # Read the text
        text_list = [item["text"] for item in texts]
        
        # Generate the embedding
        embeddings = self.encode(text_list)
        
        # Assemble the embedded data and original data
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            result = {
                "embedding": embedding.tolist(),
                "text": text["text"],
                "metadata": text.get("metadata", {})
            }
            results.append(result)
            
        return results
            
    """
    Calculation the similarties between input query and docs in the embedding file     
    Args:
        query_embedding: Input Query's embedding
        doc_embeddings: Docs in the embedding file       
    Returns: Similarties Score
    """
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        # Make sure the shape is correct
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
            
        # Using the cosine Similarties
        similarity_scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarity_scores

    """
    Store the embedded file to local file system
    Args: 
        embeddings: List that contain embedding data and original data
        output_file: Output file path
    Returns: N/A
    """
    def save_embeddings(self, embeddings: List[dict], output_file: str):
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
           
    """
    Load the stred embedding file
    Args: input_file: Input file path
    Returns: List that contain embedding data and original data
    """         
    def load_embeddings(self, input_file: str) -> List[dict]:
        import json
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

def main():
    # Run the embedding models
    embedder = TextEmbedder(chunk_size=300, chunk_overlap=50)
    
    # List of test query with different languages
    test_texts = [
        "人工智能是计算机科学的一个分支",
        "Artificial Intelligence is a branch of computer science",
        "AI is transforming the world",
        "AI正在改变世界",
        "机器学习是AI的核心技术",
        "Machine learning is the core technology of AI" ]
    
    # Generate the final embedding text use for following tasks
    print("\nGenerating embeddings for test texts...")
    embeddings = embedder.encode(test_texts)
    
    # Demo Query
    queries = [
        "什么是AI",  # Chinese demo case
        "What is AI",  # English demo case
        "AI的核心技术",  # CHinese demo case
        "core technology of AI"  # English Demo Case
    ]
    
    print("\nCross-language matching results:")
    for query in queries:
        print(f"\nQuery: {query}")
        query_embedding = embedder.encode(query)
        similarities = embedder.compute_similarity(query_embedding, embeddings)
        
        # 找到最相似的文本
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        for idx in top_indices:
            print(f"Text: {test_texts[idx]}")
            print(f"Similarity: {similarities[idx]:.4f}")
            print("---")

if __name__ == "__main__":
    main() 
