from rag_agent import init_pinecone, search_pinecone, process_search_results, generate_response, translate_query, load_merged_data
import os
from dotenv import load_dotenv

def test_rag_agent():
    # Load environment variables
    load_dotenv()
    
    # Test Pinecone initialization
    print("Initializing Pinecone...")
    try:
        index = init_pinecone()
        print("Pinecone initialization successful!")
    except Exception as e:
        print(f"Pinecone initialization failed: {str(e)}")
        return
    
    # Load content mapping
    print("Loading content mapping...")
    try:
        content_mapping = load_merged_data('merged_data.json')
        print("Content mapping loaded successfully!")
    except Exception as e:
        print(f"Failed to load content mapping: {str(e)}")
        return
    
    # Translate query
    print("Translating query...")
    chinese_query = "日本旅游"
    english_query = translate_query(chinese_query)
    print(f"Chinese query: {chinese_query}")
    print(f"English query: {english_query}")
    
    # Search with both Chinese and English queries
    zh_results = []
    en_results = []
    
    # Chinese query
    print("\nSearching for Chinese query...")
    try:
        zh_search_results = search_pinecone(index, chinese_query)
        print(f"Found {len(zh_search_results)} Chinese results")
        processed_zh_results = process_search_results(chinese_query, zh_search_results, content_mapping)
        # Filter Chinese results
        for result in processed_zh_results:
            if not result['is_english']:
                zh_results.append(result)
    except Exception as e:
        print(f"Chinese search failed: {str(e)}")
    
    # English query
    print("\nSearching for English query...")
    try:
        en_search_results = search_pinecone(index, english_query)
        print(f"Found {len(en_search_results)} English results")
        processed_en_results = process_search_results(english_query, en_search_results, content_mapping)
        # Filter English results
        for result in processed_en_results:
            if result['is_english']:
                en_results.append(result)
    except Exception as e:
        print(f"English search failed: {str(e)}")
    
    # Merge and sort results
    final_results = []
    seen_titles = set()
    
    # Add Chinese results (up to 5)
    for result in zh_results[:5]:
        if result['title'] not in seen_titles:
            final_results.append(result)
            seen_titles.add(result['title'])
    
    # Add English results (up to 5)
    for result in en_results:
        if len([r for r in final_results if r['is_english']]) >= 5:  # Reached 5 English results
            break
        if result['title'] not in seen_titles:
            final_results.append(result)
            seen_titles.add(result['title'])
    
    # Generate response
    print("\nGenerating response...")
    try:
        response = generate_response(chinese_query, final_results)
        print("\nFinal Response:")
        print(response)
    except Exception as e:
        print(f"Response generation failed: {str(e)}")
        return

if __name__ == "__main__":
    test_rag_agent() 