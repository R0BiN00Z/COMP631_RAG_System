# Travel Assistant RAG System

A bilingual (Chinese-English) travel information retrieval system using RAG (Retrieval-Augmented Generation) technology.

## Features

- Bilingual search support (Chinese and English)
- Advanced text embedding using BAAI/bge-m3 model
- Vector database storage with Pinecone
- Content summarization for English articles
- Interactive web interface with Streamlit
- Intelligent response generation using GPT-4

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd IRnew
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

1. Start the web interface:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter your travel-related questions in either Chinese or English

## Project Structure

- `app.py`: Streamlit web interface
- `rag_api.py`: Main RAG system implementation
- `rag_agent.py`: Core RAG functionality
- `embedding.py`: Text embedding and processing
- `text_summarizer.py`: Content summarization
- `test_rag_agent.py`: Test script for RAG functionality

## Dependencies

- PyTorch 2.6.0
- Sentence Transformers 4.1.0
- Pinecone 6.0.2
- OpenAI 1.75.0
- LangChain 0.3.23
- Streamlit 1.32.0
- And other dependencies listed in requirements.txt

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 