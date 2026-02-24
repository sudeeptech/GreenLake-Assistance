from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("💬 GreenLake Assist")

# -------------------------
# Load and prepare documents (RUN ONLY ONCE)
# -------------------------
@st.cache_resource
def load_vector_store():

    # load document
    loader = TextLoader("sample.txt")
    docs = loader.load()

    # split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # embeddings (free local embeddings)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # create vector DB
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore

vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------
# Custom prompt (IMPORTANT)
# → answers only from document
# -------------------------
prompt_template = """
You are a helpful assistant.

Use ONLY the provided context to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------------
# RAG chain
# -------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------
# Chat history
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User input
# -------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # RAG response
    response = qa_chain.run(user_prompt)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
