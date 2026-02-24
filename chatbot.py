from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

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
# reload document when file updated
if st.button("🔄 Reload Document"):
    st.cache_resource.clear()
    st.rerun()

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# display chat history
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
# RAG SETUP (runs once)
# -------------------------
@st.cache_resource
def setup_rag():

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # load document
    loader = TextLoader("sample.txt")  # make sure file exists
    docs = loader.load()

    # split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # create embeddings (free local model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # create vector database
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


# initialize retriever
retriever = setup_rag()

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask from document...")

if user_prompt:

    # show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # retrieve relevant document chunks
    docs = retriever.invoke(user_prompt)

    # combine context
    context = "\n".join([doc.page_content for doc in docs])

    # RAG prompt (prevents hallucination)
    rag_prompt = f"""
You are an internal company support assistant.

Follow these rules strictly:

1. Answer ONLY from the provided context.
2. If answer is not available → say "I don't know".
3. Give clear, simple, user-friendly explanations.
4. Format answers in steps or bullet points when possible.
5. Use professional but easy language.
6. Do not copy text directly — explain in your own words.
7. Keep answers structured and helpful for employees.

Context:
{context}

User Question:
{user_prompt}

Helpful Answer:
"""

    # generate response
    response = llm.invoke(rag_prompt)
    assistant_response = response.content

    # save response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    # display response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
