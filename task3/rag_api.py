# rag_api.py
from rag_agent import Initializer, Config, SearchEngine, TextEmbedder, ContentProcessor, QueryTranslator, ResponseGenerator, DataLoader

class RAGAgent:
    def __init__(self):
        # 初始化所有必要组件
        self.config = Config()
        self.index = Initializer.init_pinecone(self.config)
        self.openai_client = Initializer.init_openai(self.config)
        self.summarizer = Initializer.init_summarizer(self.config)
        
        # 初始化处理器
        self.search_engine = SearchEngine(self.index, TextEmbedder())
        self.content_processor = ContentProcessor(self.summarizer)
        self.query_translator = QueryTranslator(self.openai_client)
        self.response_generator = ResponseGenerator(self.openai_client)
        
        # 加载数据
        self.merged_data = DataLoader.load_merged_data('merged_data.json')

    def get_response(self, query: str) -> str:
        try:
            # Translate query
            print("\nTranslating query...")
            queries = self.query_translator.translate_query(query)
            print(f"Chinese query: {queries['zh']}")
            print(f"English query: {queries['en']}")

            # Search and process results
            print("\nSearching and processing results...")
            zh_results = self.search_engine.search_pinecone(queries['zh'])
            en_results = self.search_engine.search_pinecone(queries['en'])

            # Process content
            processed_zh = self.content_processor.process_results(zh_results, self.merged_data)
            processed_en = self.content_processor.process_results(en_results, self.merged_data)

            # Rerank results
            reranked_zh = self.content_processor.rerank_results(processed_zh, queries['zh'])
            reranked_en = self.content_processor.rerank_results(processed_en, queries['en'])

            # Take top results
            final_results = reranked_zh[:4] + reranked_en[:1]

            # Print final results details
            print("\nFinal matched results:")
            for i, result in enumerate(final_results, 1):
                print(f"\nResult {i}:")
                print(f"Title: {result['title']}")
                print(f"Language: {'English' if result.get('is_english') else 'Chinese'}")
                print(f"Vector Score: {result.get('vector_score', 0):.4f}")
                print(f"Final Score: {result.get('final_score', 0):.4f}")
                print(f"Content length: {len(result.get('content', ''))}")
                print("-" * 50)

            print("Summarizing content that is too long...")
            # Summarize English content
            final_results = self.content_processor.summarize_english_content(final_results)

            # Generate response
            response = self.response_generator.generate_response(query, final_results)
            return response

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"