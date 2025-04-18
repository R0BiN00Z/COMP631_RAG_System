import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

load_dotenv()            # Load the Environment

class TextSummarizer:
    """
    Initialize text summarizer
    Args:
        api_key: Gemini API key, if None will get from environment variable
        max_workers: Maximum number of threads for parallel processing
        batch_size: Batch size for processing
    Return: N/A
    """
    def __init__(self, api_key: str = None, max_workers: int = 4, batch_size: int = 5):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please provide Gemini API key or set GEMINI_API_KEY environment variable")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=2048,    # Increase output length limit
            top_p=0.8,                 # Adjust sampling parameter
            top_k=40                   # Adjust sampling parameter
        )
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialize text splitter with adjusted chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,      # Reduce chunk size
            chunk_overlap=100,    # Reduce overlap
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]  # Optimize separators
        )
        
        # Create more concise Map phase prompt template
        map_template = """Please summarize the key information of the following text in concise language:
{context}

Key information:"""
        self.map_prompt = PromptTemplate.from_template(map_template)
        
        # Create more concise Reduce phase prompt template
        reduce_template = """Please combine the following summaries into one concise complete summary:
{context}

Combined summary:"""
        self.reduce_prompt = PromptTemplate.from_template(reduce_template)
        
        # Create Map and Reduce Chain
        self.map_chain = (RunnablePassthrough() | self.map_prompt | self.llm)
        self.reduce_chain = (RunnablePassthrough() | self.reduce_prompt | self.llm)

    def _process_batch(self, batch: List[Document]) -> List[str]:
        try:
            # Process all documents in one batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        lambda doc: self.map_chain.invoke({"context": doc.page_content}).content,
                        doc
                    )
                    for doc in batch
                ]
                return [future.result() for future in as_completed(futures)]
        except Exception as e:
            print(f"Error processing batch: {e}")
            return []

    def _batch_documents(self, docs: List[Document]) -> List[List[Document]]:
        it = iter(docs)
        return list(iter(lambda: list(islice(it, self.batch_size)), []))

    """
    Summary the long text
    Args: text: Input long text
    Returns: str: Summary Result
    """
    def summarize(self, text: str) -> str:
        # Split text
        docs = self.text_splitter.create_documents([text])
        print(f"Text split into {len(docs)} chunks")
        
        # Split documents into batches
        batches = self._batch_documents(docs)
        print(f"Split into {len(batches)} batches for processing")
        
        # Process all batches
        all_summaries = []
        for i, batch in enumerate(batches, 1):
            print(f"Processing batch {i}/{len(batches)}...")
            summaries = self._process_batch(batch)
            all_summaries.extend(summaries)
        
        # Combine all summaries
        print("Combining all summaries...")
        combined_text = "\n".join(all_summaries)
        final_summary = self.reduce_chain.invoke({"context": combined_text}).content
        
        return final_summary

def main():
    # Sample Run, each batch contain 5 documents and 4 threads were being assigned to this task
    api_key = os.getenv("GEMINI_API_KEY")
    summarizer = TextSummarizer(api_key, max_workers=4, batch_size=5)
    
    # Read data from merged_data.json
    with open('merged_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        chengdu_content = None
        for item in data:
            if item.get('title') == 'Chengdu':
                chengdu_content = item.get('content')
                break
        
        if not chengdu_content:
            print("No content found with title 'Chengdu'")
            return
    
    # Generate summary
    print("Processing text...")
    summary = summarizer.summarize(chengdu_content)
    print("\nSummary result:")
    print(summary)

if __name__ == "__main__":
    main() 
