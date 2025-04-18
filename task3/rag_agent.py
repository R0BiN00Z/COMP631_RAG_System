import argparse
import json
import numpy as np
from typing import List, Dict, Tuple
from embedding import TextEmbedder
from pinecone import Pinecone
from tqdm import tqdm
from dotenv import load_dotenv
import os
from openai import OpenAI
from text_summarizer import TextSummarizer
from googletrans import Translator

# ============= Configuration and Initialization =============
class Config:
    def __init__(self):
        load_dotenv()
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not all([self.pinecone_api_key, self.openai_api_key, self.gemini_api_key]):
            raise ValueError("Missing required API keys in .env file")

class Initializer:
    @staticmethod
    def init_pinecone(config: Config) -> any:
        pc = Pinecone(api_key=config.pinecone_api_key)
        return pc.Index("irnew")
    
    @staticmethod
    def init_openai(config: Config) -> OpenAI:
        return OpenAI(api_key=config.openai_api_key)
    
    @staticmethod
    def init_summarizer(config: Config) -> TextSummarizer:
        """Initialize text summarizer"""
        return TextSummarizer(api_key=config.gemini_api_key)

# ============= Data Loading and Processing =============
class DataLoader:
    @staticmethod
    def load_merged_data(file_path: str) -> List[Dict]:
        print(f"Loading merged data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} documents")
            return data

# ============= Search Functions =============
class SearchEngine:
    def __init__(self, index, embedder: TextEmbedder):
        self.index = index
        self.embedder = embedder

    def search_pinecone(self, query: str, top_k: int = 20) -> List[Dict]:
        query_vector = self.embedder.encode(query)
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return self._process_search_results(results)

    def _process_search_results(self, results: Dict) -> List[Dict]:
        processed_results = []
        for match in results['matches']:
            processed_results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata'],
                'values': match.get('values', [])
            })
        return processed_results

# ============= Content Processing =============
class ContentProcessor:
    def __init__(self, summarizer: TextSummarizer):
        self.summarizer = summarizer

    def process_results(self, search_results: List[Dict], merged_data: List[Dict]) -> List[Dict]:
        processed_results = []
        for result in search_results:
            processed_result = self._process_single_result(result, merged_data)
            if processed_result:
                processed_results.append(processed_result)
        return processed_results

    def _process_single_result(self, result: Dict, merged_data: List[Dict]) -> Dict:
        title = result.get('metadata', {}).get('title', '')
        if not title:
            return None
            
        for doc in merged_data:
            if doc.get('title') == title:
                return {
                    'title': title,
                    'content': doc.get('content', ''),
                    'score': result.get('score', 0),
                    'is_english': result.get('metadata', {}).get('language') == 'en'
                }
        return None

    def summarize_english_content(self, results: List[Dict]) -> List[Dict]:
        for result in results:
            print(result['is_english'])
            if result['is_english']:
                
                try:
                    result['content'] = self.summarizer.summarize(result['content'])
                except Exception as e:
                    print(f"Error summarizing content: {str(e)}")
        return results

    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """重新排序结果，考虑多个因素"""
        seen_titles = set()  # 用于去重的集合
        reranked_results = []
        
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            if result['title'] not in seen_titles:  # 检查标题是否重复
                # 计算分数
                vector_score = result['score'] * 0.6
                title_score = self._calculate_keyword_match(result['title'], query) * 0.25
                content_score = self._calculate_keyword_match(result['content'][:1000], query) * 0.15
                
                result['vector_score'] = vector_score
                result['title_score'] = title_score
                result['content_score'] = content_score
                result['final_score'] = vector_score + title_score + content_score
                
                reranked_results.append(result)
                seen_titles.add(result['title'])  # 将标题添加到已见集合
        
        return sorted(reranked_results, key=lambda x: x['final_score'], reverse=True)

    def _calculate_keyword_match(self, text: str, query: str) -> float:
        """计算关键词匹配度"""
        if not text or not query:
            return 0.0
            
        text_lower = text.lower()
        query_words = set(query.lower().split())
        
        # 计算查询词在文本中的出现比例
        matched_words = sum(1 for word in query_words if word in text_lower)
        return matched_words / len(query_words) if query_words else 0.0

# ============= Query Translation =============
class QueryTranslator:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def translate_query(self, query: str) -> Dict[str, str]:
        try:
            prompt = self._create_translation_prompt(query)
            response = self._get_gpt_response(prompt)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error translating query: {str(e)}")
            return {'zh': query, 'en': query}

    def _create_translation_prompt(self, query: str) -> str:
        return f"""You are a search query optimization expert. Please generate optimized search queries in both Chinese and English for the following travel-related query.

Original query: {query}

Requirements:
- Focus on cities, attractions, and experiences
- Include popular destinations and landmarks
- Add relevant cultural and culinary experiences
- DO NOT include time-related information
- Make the queries natural and search-engine friendly
- Return the results in JSON format with 'zh' and 'en' keys"""

    def _get_gpt_response(self, prompt: str):
        return self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a search query optimization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

# ============= Response Generation =============
class ResponseGenerator:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        if not search_results:
            return "Sorry, no relevant results found."
            
        try:
            prompt = self._create_response_prompt(query, search_results)
            response = self._get_gpt_response(prompt)
            print(f"Generated response: {response.choices[0].message.content}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(query, search_results)

    def _create_response_prompt(self, query: str, results: List[Dict]) -> str:
        formatted_results = self._format_results(results)
        return f"""You are a professional travel advisor. Please answer the user's query based on the following search results.

User query: {query}

Search results:
{formatted_results}

Please generate a structured response in English including:
1. Recommended attractions and routes
2. Food recommendations
3. Practical tips

In English!
use some emojis to make the response more engaging"""

    def _format_results(self, results: List[Dict]) -> str:
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. Title: {result['title']}\n   Content: {result['content']}\n   Relevance: {result['score']:.4f}"
            )
        return "\n".join(formatted)

    def _generate_fallback_response(self, query: str, results: List[Dict]) -> str:
        # 保持原有的fallback逻辑
        pass

    def _get_gpt_response(self, prompt: str):
        return self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a professional travel advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

# ============= Main Function =============
def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='RAG Agent for Tourism Information')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    args = parser.parse_args()

    # 初始化配置和组件
    config = Config()
    index = Initializer.init_pinecone(config)
    openai_client = Initializer.init_openai(config)
    summarizer = Initializer.init_summarizer(config)

    # 初始化各个处理器
    search_engine = SearchEngine(index, TextEmbedder())
    content_processor = ContentProcessor(summarizer)
    query_translator = QueryTranslator(openai_client)
    response_generator = ResponseGenerator(openai_client)

    # 查询翻译
    queries = query_translator.translate_query(args.query)
    print(f"Chinese query: {queries['zh']}")
    print(f"English query: {queries['en']}")

    # 加载数据
    merged_data = DataLoader.load_merged_data('merged_data.json')

    # 搜索和处理结果
    zh_results = search_engine.search_pinecone(queries['zh'])
    en_results = search_engine.search_pinecone(queries['en'])

    # 处理内容
    processed_zh = content_processor.process_results(zh_results, merged_data)
    processed_en = content_processor.process_results(en_results, merged_data)
 # 确保查询是字符串类型
    zh_query = str(queries.get('zh', args.query))
    en_query = str(queries.get('en', args.query))
    
    print(f"Chinese query: {zh_query}")
    print(f"English query: {en_query}")
    # 重新排序结果
    reranked_zh = content_processor.rerank_results(processed_zh, queries['zh'])
    reranked_en = content_processor.rerank_results(processed_en, queries['en'])

    # 取排序后的前10条结果
    final_results = reranked_zh[:5] + reranked_en[:5]

    # 为英文内容生成摘要
    final_results = content_processor.summarize_english_content(final_results)

    # 生成响应
    response = response_generator.generate_response(args.query, final_results)
    print("\nGenerated response:")
    print(response)

if __name__ == "__main__":
    main() 