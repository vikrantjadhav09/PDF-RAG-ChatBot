import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# STEP 1 — Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")
    st.stop()


# ============================================================
# STEP 2 — Streamlit UI
# ============================================================

st.title("📄 PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions based on its content."
)


# ============================================================
# STEP 3 — Upload PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    # ========================================================
    # STEP 4 — Save uploaded PDF
    # ========================================================

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())


    # ========================================================
    # STEP 5 — Load PDF
    # ========================================================

    loader = PyPDFLoader("temp.pdf")

    documents = loader.load()

    st.write(
        f"📄 Pages loaded: {len(documents)}"
    )


    # ========================================================
    # STEP 6 — Split documents into chunks
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(
    documents
        )

    st.write(
    f"🧩 Chunks created: {len(chunks)}"
    )

    if not chunks:
        st.error(
        "❌ No readable text was found in this PDF. "
        "This may be a scanned/image-based PDF. "
        "Please upload a PDF containing selectable text."
        )
        st.stop()


    # ========================================================
    # STEP 7 — Create Embeddings
    # ========================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success(
        "✅ HuggingFace Embeddings created!"
    )


    # ========================================================
    # STEP 8 — Create FAISS Vector Database
    # ========================================================

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.success(
        "✅ FAISS vector database created!"
    )


    # ========================================================
    # STEP 9 — Create Retriever
    # ========================================================

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    st.success(
        "✅ Retriever created!"
    )


    # ========================================================
    # STEP 10 — Create LLM
    # ========================================================

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=GROQ_API_KEY
    )


    st.success(
        "✅ Groq LLM created!"
    )


    # ========================================================
    # STEP 11 — Create Prompt
    # ========================================================

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful PDF question-answering assistant.

        Answer the question using ONLY the provided context.

        If the answer is not present in the context,
        say exactly:

        "I don't know based on the provided PDF."

        Do not use outside knowledge.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )


    # ========================================================
    # STEP 12 — Create LangChain RAG Chain
    # ========================================================

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    # ========================================================
    # Ask Question
    # ========================================================

    question = st.text_input(
        "🔎 Ask a question about your PDF:"
    )


    if question:

        with st.spinner("🤖 Thinking..."):

            answer = rag_chain.invoke(
                question
            )


        st.write("### 🤖 RAG Answer")

        st.write(answer)