# chatbot.py

import os
from dotenv import load_dotenv
import streamlit as st

# Groq LLM
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Embeddings & Document Loading
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()
# Ensure GROQ_API_KEY is set in .env or environment
# GROQ_API_KEY=your_key_here

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
# LLM INIT (Groq) safely
# -------------------------
@st.cache_resource
def get_llm():
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = get_llm()

# -------------------------
# SIMPLE PYTHON TEXT SPLITTER
# -------------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------
# RAG SETUP (in-memory) with loading spinner
# -------------------------
@st.cache_resource
def setup_rag():
    retriever = []
    try:
        with st.spinner("Loading document and generating embeddings..."):
            loader = TextLoader("sample.txt")
            docs = loader.load()

            split_docs = []
            for doc in docs:
                chunks = split_text(doc.page_content, chunk_size=500, overlap=50)
                for chunk in chunks:
                    split_docs.append({"page_content": chunk})

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            retriever = [{"doc": doc, "embedding": embeddings.embed_query(doc["page_content"])} for doc in split_docs]

    except FileNotFoundError:
        st.error("Error: sample.txt not found!")
    except Exception as e:
        st.error(f"Error loading document: {e}")

    return retriever

retriever = setup_rag()

def simple_retrieve(query):
    """Return top 3 document chunks (small dataset)"""
    if not retriever:
        return []
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
    if docs:
        context = "\n".join([doc["page_content"] for doc in docs])
    else:
        context = ""

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

    # -------------------------
    # Call LLM safely using Groq .generate()
    # -------------------------
    if llm:
        try:
            messages = [HumanMessage(content=rag_prompt)]
            completion = llm.generate(messages)
            assistant_response = completion.generations[0][0].text
            if not assistant_response.strip():
                assistant_response = "I don't know."
        except Exception as e:
            st.error(f"Error generating response: {e}")
            assistant_response = "I don't know."
    else:
        assistant_response = "LLM not initialized. Cannot answer."

    # Save and display assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
