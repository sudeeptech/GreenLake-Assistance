# chatbot.py

import os
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()

# -------------------------
# STREAMLIT PAGE SETUP
# -------------------------
st.set_page_config(
    page_title="GreenLake Assist",
    page_icon="🤖",
    layout="centered",
)
st.title("💬 GreenLake Assist (RAG)")

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# LLM INIT (Groq)
# -------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
)

# -------------------------
# SIMPLE PYTHON TEXT SPLITTER
# -------------------------
def split_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------
# RAG SETUP (in-memory)
# -------------------------
@st.cache_resource
def setup_rag():
    # Try to load the document
    try:
        loader = TextLoader("sample.txt")  # Ensure sample.txt exists
        docs = loader.load()
    except FileNotFoundError:
        print("Error: sample.txt not found!")
        return []
    except Exception as e:
        print(f"Error loading document: {e}")
        return []

    # Split documents into chunks
    split_docs = []
    for doc in docs:
        chunks = split_text(doc.page_content, chunk_size=500, overlap=50)
        for chunk in chunks:
            split_docs.append({"page_content": chunk})

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # In-memory retriever
    retriever = [{"doc": doc, "embedding": embeddings.embed_query(doc["page_content"])} for doc in split_docs]

    return retriever

retriever = setup_rag()

def simple_retrieve(query):
    # Return top 3 documents (for small datasets)
    return [item["doc"] for item in retriever][:3]

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Retrieve relevant docs
    docs = simple_retrieve(user_prompt)
    context = "\n".join([doc["page_content"] for doc in docs])

    # RAG prompt
    rag_prompt = f"""
You are an internal company support assistant.

Follow these rules strictly:
1. Answer ONLY from the provided context.
2. If answer is not available → say "I don't know".
3. Give clear, simple, user-friendly explanations.
4. Use bullet points when helpful.

Context:
{context}

User Question:
{user_prompt}

Helpful Answer:
"""

    # Get assistant response
    assistant_response = llm.predict(rag_prompt)

    # Save and display assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
