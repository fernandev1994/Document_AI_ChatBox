import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Set OpenRouter key and base URL
# OpenRouter key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
# Hugging Face for embeddings (free)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Use GPT-3.5 from OpenRouter
# MUST use OpenRouter-compatible model ID
Settings.llm = OpenAI(model="gpt-3.5-turbo-0613")

# UI
st.title("📄 Ask AI About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Reading and indexing your document..."):
        reader = SimpleDirectoryReader(input_files=["temp.pdf"])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)

    # Initialize session state if not already present
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history (questions and responses)
    for chat in st.session_state.history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")

        # Create an empty container to hold the input field at the bottom
    input_container = st.empty()  # This creates a placeholder for the input field

    query = input_container.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Thinking..."):
            response = index.as_query_engine().query(query)
            # Save query and response in session state for chat history
            st.session_state.history.append(
                {"question": query, "answer": response.response})
            st.write("**Answer:**", response.response)
