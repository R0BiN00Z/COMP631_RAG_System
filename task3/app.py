# app.py
import streamlit as st
from rag_api import RAGAgent

def main():
    st.title("Travel Assistant")
    
    # Initialize RAG Agent (if not already initialized)
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = RAGAgent()
    
    # User input
    query = st.text_input("Please enter your travel-related question:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching for relevant information..."):
                try:
                    # Get response
                    response = st.session_state.rag_agent.get_response(query)
                    
                    # Display response
                    st.write("### Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()