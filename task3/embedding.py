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
    
    def detect_language(self, text: str) -> str:
        """
        检测文本的主要语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码 ('zh' 或 'en')
        """
        # 简单的语言检测
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars / len(text) > 0.3:  # 如果中文字符占比超过30%
            return 'zh'
        return 'en'
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        将文本分割成多个块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        # 检测语言
        lang = self.detect_language(text)
        
        if lang == 'zh':
            # 中文分词
            words = list(jieba.cut(text))
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > self.chunk_size:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                        # 保留重叠部分
                        overlap = int(self.chunk_overlap / 2)
                        current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                        current_length = sum(len(w) for w in current_chunk)
                
                current_chunk.append(word)
                current_length += len(word)
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
        
        else:
            # 英文按空格分割
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > self.chunk_size:  # +1 for space
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # 保留重叠部分
                        overlap = int(self.chunk_overlap / 2)
                        current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                        current_length = sum(len(w) + 1 for w in current_chunk) - 1
                
                current_chunk.append(word)
                current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def encode(self, text: str, return_chunks: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """
        生成文本的嵌入向量
        
        Args:
            text: 输入文本
            return_chunks: 是否返回分块信息
            
        Returns:
            如果 return_chunks 为 False，返回嵌入向量
            如果 return_chunks 为 True，返回 (嵌入向量, 分块信息)
        """
        # 检测语言
        language = self.detect_language(text)
        
        # 分块处理
        chunks = self.split_into_chunks(text)
        
        # 生成嵌入向量
        embeddings = []
        chunk_info = []
        
        for i, chunk in enumerate(chunks):
            # 将文本转换为模型输入
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成嵌入向量
            with torch.no_grad():
                output = self.model(encoded)
                embedding = output['sentence_embedding'].cpu().numpy()
            
            embeddings.append(embedding)
            chunk_info.append((0, i))  # 文档索引始终为0，因为我们只处理单个文档
        
        # 合并所有块的嵌入向量
        final_embedding = np.mean(embeddings, axis=0)
        
        if return_chunks:
            return final_embedding, chunk_info
        return final_embedding
    
    def encode_with_metadata(self, texts: List[dict]) -> List[dict]:
        """
        对带有元数据的文本进行编码
        
        Args:
            texts: 包含文本和元数据的字典列表
                  [{"text": "...", "metadata": {...}}, ...]
                  
        Returns:
            包含嵌入向量和元数据的字典列表
        """
        # 提取文本
        text_list = [item["text"] for item in texts]
        
        # 生成嵌入向量
        embeddings = self.encode(text_list)
        
        # 合并嵌入向量和元数据
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            result = {
                "embedding": embedding.tolist(),
                "text": text["text"],
                "metadata": text.get("metadata", {})
            }
            results.append(result)
            
        return results
    
    def compute_similarity(self, 
                         query_embedding: np.ndarray,
                         doc_embeddings: np.ndarray) -> np.ndarray:
        """
        计算查询向量和文档向量之间的相似度
        
        Args:
            query_embedding: 查询文本的嵌入向量
            doc_embeddings: 文档文本的嵌入向量列表
            
        Returns:
            相似度分数数组
        """
        # 确保维度正确
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
            
        # 使用余弦相似度
        similarity_scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarity_scores
    
    def save_embeddings(self, 
                       embeddings: List[dict],
                       output_file: str):
        """
        保存嵌入向量到文件
        
        Args:
            embeddings: 包含嵌入向量和元数据的列表
            output_file: 输出文件路径
        """
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
            
    def load_embeddings(self, input_file: str) -> List[dict]:
        """
        从文件加载嵌入向量
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            包含嵌入向量和元数据的列表
        """
        import json
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

def main():
    # 测试代码
    embedder = TextEmbedder(chunk_size=300, chunk_overlap=50)
    
    # 测试跨语言匹配
    test_texts = [
        "人工智能是计算机科学的一个分支",  # 中文
        "Artificial Intelligence is a branch of computer science",  # 英文
        "AI is transforming the world",  # 英文
        "AI正在改变世界",  # 中文
        "机器学习是AI的核心技术",  # 中文
        "Machine learning is the core technology of AI"  # 英文
    ]
    
    # 生成嵌入向量
    print("\nGenerating embeddings for test texts...")
    embeddings = embedder.encode(test_texts)
    
    # 测试查询
    queries = [
        "什么是AI",  # 中文查询
        "What is AI",  # 英文查询
        "AI的核心技术",  # 中文查询
        "core technology of AI"  # 英文查询
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