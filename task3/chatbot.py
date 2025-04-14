from sentence_transformers import SentenceTransformer
from datasets import Dataset
import numpy as np
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

logging.basicConfig(level=logging.INFO)

# Reuse the ChunkedRetriever from Task 2
class ChunkedRetriever:
    def __init__(self, corpus_path, chunk_size=300):
        # Load the model and parameters
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.chunk_size = chunk_size

        # Load and process dataset
        self.dataset = Dataset.from_csv(corpus_path)
        self.chunks = []
        self.chunk_meta = []
        self.doc_map = {}

        # Process each document into chunks
        for idx, doc in enumerate(self.dataset):
            doc_id = doc.get('id', f'doc{idx}')
            title = doc.get('title', '')
            text = doc.get('text', '')

            # Ensure text fields are strings
            if not isinstance(text, str):
                text = str(text)
            if not isinstance(title, str):
                title = str(title)

            # Store original document content
            self.doc_map[doc_id] = {
                'title': title,
                'text': text
            }

            # Split document into chunks
            for i in range(0, len(text), self.chunk_size):
                chunk_text = text[i:i + self.chunk_size]
                full_chunk = f"{title} {chunk_text}".strip()
                self.chunks.append(full_chunk)
                self.chunk_meta.append({
                    "doc_id": doc_id,
                    "chunk_index": i // self.chunk_size,
                    "chunk_text": chunk_text
                })

        # Generate embeddings for all chunks
        self.chunk_embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        logging.info(f"Initialized retriever with {len(self.chunks)} chunks from {len(self.doc_map)} documents")

    def query(self, queries, top_k=5, expansion_factor=10):
        if not isinstance(queries, dict):
            raise ValueError("Queries must be a {qid: text} dictionary")

        # Encode all queries
        query_texts = list(queries.values())
        query_embs = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate similarity scores
        scores = np.dot(query_embs, self.chunk_embeddings.T)

        results = {}
        for idx, qid in enumerate(queries.keys()):
            # Get expanded list of candidate chunks
            sorted_indices = np.argsort(scores[idx])[::-1][:top_k * expansion_factor]

            seen_docs = set()
            results_for_query = []

            # Deduplicate documents and select top results
            for i in sorted_indices:
                meta = self.chunk_meta[i]
                doc_id = meta["doc_id"]
                score = float(scores[idx][i])

                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    doc = self.doc_map[doc_id]
                    results_for_query.append({
                        "doc_id": doc_id,
                        "score": score,
                        "title": doc["title"],
                        "text": doc["text"]
                    })
                if len(results_for_query) >= top_k:
                    break
            results[qid] = results_for_query

        return results

# RAG Chatbot that uses the ChunkedRetriever
class RAGChatbot:
    def __init__(self, retriever, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cpu"):
        self.retriever = retriever
        logging.info(f"Loading LLM: {model_name}")
        
        # Use Qwen2.5-1.5B-Instruct model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with appropriate settings for CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use regular 32-bit precision for CPU
            trust_remote_code=True
        )
        
        logging.info("LLM loaded successfully")
        
    def format_prompt(self, query, retrieved_docs, domain_expertise="information retrieval specialist"):
        """Format the prompt for the Qwen model with retrieved documents as context"""
        # Qwen uses a specific chat format
        system_prompt = f"You are a helpful {domain_expertise}. Answer the user's question based only on the provided documents."
        
        # Format the retrieved documents as context
        context = "Here are the relevant documents:\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc['text']
            if len(doc_text) > 500:  # Truncate very long documents
                doc_text = doc_text[:500] + "..."
            
            context += f"Document {i}: {doc['title']}\n{doc_text}\n\n"
        
        # Combine into the format expected by Qwen's tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context + f"\nQuestion: {query}"}
        ]
        
        return messages
        
    def generate_answer(self, messages, max_length=512):
        """Generate a response from the LLM using the formatted prompt"""
        # Use Qwen's chat format
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate with appropriate parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode and return the generated text
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated answer (model response part)
        answer = decoded.split("assistant")[-1].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
        
        return answer
    
    def answer_query(self, query, top_k=3):
        """Main method to answer a user query using RAG"""
        # Retrieve relevant documents
        logging.info(f"Retrieving documents for query: {query}")
        results = self.retriever.query({"user_query": query}, top_k=top_k)
        retrieved_docs = results["user_query"]
        
        # Format prompt with retrieved documents
        messages = self.format_prompt(query, retrieved_docs)
        
        # Generate answer using the LLM
        logging.info("Generating answer with LLM")
        answer = self.generate_answer(messages)
        
        # If the answer is empty or too short, provide a fallback response
        if not answer or len(answer) < 10:
            answer = "Based on the retrieved documents, I can't generate a specific answer. Please try rephrasing your question."
        
        return answer, retrieved_docs

# Create Gradio interface
def create_chatbot_interface(rag_chatbot):
    def respond(message, history):
        answer, docs = rag_chatbot.answer_query(message)
        
        # Add citation information
        sources = "\n\nSources:\n" + "\n".join([f"- {doc['title']} (Score: {doc['score']:.3f})" for doc in docs])
        
        return answer + sources
    
    # Create a simple chat interface
    demo = gr.ChatInterface(
        respond,
        title="Information Retrieval RAG Chatbot",
        description="Ask questions about the corpus and get answers with citations.",
        examples=["What information can you provide about this topic?", 
                 "Can you summarize the main points?",
                 "What are the key concepts discussed?"]
    )
    
    return demo

# Main function to run the application
def main(corpus_path="./huggingface_corpus.csv"):
    # Initialize the retriever
    logging.info("Initializing retriever...")
    retriever = ChunkedRetriever(corpus_path)
    
    # Initialize the RAG chatbot
    logging.info("Initializing RAG chatbot...")
    rag_chatbot = RAGChatbot(retriever)
    
    # Create and launch the Gradio interface
    logging.info("Creating interface...")
    demo = create_chatbot_interface(rag_chatbot)
    
    # Launch the interface with share=True to create a public link
    logging.info("Launching interface...")
    demo.launch(share=True)

if __name__ == "__main__":
    # Replace with the path to your corpus CSV file
    main("./huggingface_corpus_full.csv")
