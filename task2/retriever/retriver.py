from sentence_transformers import SentenceTransformer
from datasets import Dataset
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class ChunkedRetriever:
    """
    Initialize the retriever and process the corpus.

    Parameters:
        corpus_path (str): Path to CSV file containing documents with columns ['id', 'title', 'text']
        chunk_size (int, optional): Character length for document chunks. Defaults to 300.

    Raises:
        FileNotFoundError: If the corpus file cannot be loaded
    """
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

    """
    Execute semantic search queries against the corpus.

    Parameters:
        queries (dict): Dictionary of {query_id: query_text} pairs to search
        top_k (int, optional): Number of documents to return per query. Defaults to 5.
        expansion_factor (int, optional): Multiplier for initial candidate selection to improve recall.
                                          Defaults to 10.

    Returns:
        dict: {query_id: list[dict]} where each result dict contains:
            - doc_id: Document identifier
            - score: Similarity score (0-1)
            - title: Document title
            - text: Full document text

    Raises:
        ValueError: If queries parameter is not a dictionary
    """
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


# Main function used to run the retriever part code
if __name__ == "__main__":
    # Choose the data source
    retriever = ChunkedRetriever("./huggingface_corpus.csv", chunk_size=100)

    # User query input example (Could change to other, task3 will change this section code)
    results = retriever.query({
        "q1": "做个好吃的饭菜 非常喜欢",
        "q2": "上海美食"
    }, top_k=10)

    # Print the result out
    for qid, docs in results.items():
        print(f"\n==== Query: {qid} ====")
        for doc in docs:
            print(f"\nScore: {doc['score']:.3f} | Doc ID: {doc['doc_id']}")
            # print(f"Title: {doc['title']}")
            print(f"Content Preview: {doc['text'][:300]}...")