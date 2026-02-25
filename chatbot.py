from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# RAG SETUP (in-memory, FAISS-free)
# -------------------------
@st.cache_resource
def setup_rag():
    loader = TextLoader("sample.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # simple in-memory retriever
    retriever = [{"doc": doc, "embedding": embeddings.embed_query(doc.page_content)} for doc in split_docs]

    return retriever

retriever = setup_rag()

def simple_retrieve(query):
    # return top 3 documents (for small datasets)
    return [item["doc"] for item in retriever][:3]

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # retrieve context
    docs = simple_retrieve(user_prompt)
    context = "\n".join([doc.page_content for doc in docs])

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

    assistant_response = llm.predict(rag_prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
