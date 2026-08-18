# 📄 PDF RAG Chatbot

An AI-powered **PDF Question Answering Chatbot** built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload a PDF document and ask questions about its content. Instead of sending the entire document directly to the Large Language Model, the application extracts the PDF text, splits it into smaller chunks, converts those chunks into vector embeddings, stores them in a **FAISS vector database**, retrieves the most relevant chunks for a user's question, and sends the retrieved context to a **Groq-powered LLM** to generate an answer.

The application is built with **Python, Streamlit, LangChain, HuggingFace Sentence Transformers, FAISS, and Groq**.

---

## 🌐 Live Demo

**Live Application:** PDF RAG Chatbot

https://pdf-rag-chatbot-09.streamlit.app/ 

The application is deployed using **Streamlit Community Cloud**.

### Try it

1. Open the deployed application.
2. Upload a PDF document.
3. Wait for the document processing to complete.
4. Enter a question related to the uploaded PDF.
5. The RAG pipeline retrieves relevant information.
6. The LLM generates an answer based on the retrieved context.

---

# 🎯 Project Objective

Traditional document searching requires users to manually read a large document and search for relevant information.

For example, imagine a user has a **227-page Machine Learning PDF** and wants to know:

> "What is Machine Learning?"

Instead of manually searching through hundreds of pages, this application allows the user to simply ask the question.

The application automatically:

```text
PDF
 ↓
Extract Text
 ↓
Split Text into Chunks
 ↓
Generate Embeddings
 ↓
Store Vectors in FAISS
 ↓
Retrieve Relevant Chunks
 ↓
Send Context to LLM
 ↓
Generate Answer
```

This makes it easier to interact with large PDF documents using natural language.

---

# ✨ Features

* 📄 Upload PDF documents
* 🔍 Ask natural-language questions about the PDF
* 🧠 Retrieval-Augmented Generation (RAG)
* ✂️ Recursive text chunking
* 🔢 HuggingFace sentence embeddings
* ⚡ FAISS vector database
* 🔎 Semantic similarity-based retrieval
* 🤖 Groq LLM integration
* 🔗 LangChain RAG pipeline
* 🌐 Streamlit web interface
* 🔐 API key management using environment variables and Streamlit Secrets
* 🚀 Deployed on Streamlit Community Cloud
* 📊 Displays number of pages loaded
* 🧩 Displays number of chunks created
* ✅ Displays the status of major RAG components

---

# 🧠 What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

RAG combines:

```text
Information Retrieval
        +
Large Language Model
        =
Retrieval-Augmented Generation
```

Instead of asking the LLM to answer a question using only its internal knowledge, the application first retrieves relevant information from the uploaded PDF.

The retrieved information is then provided to the LLM as context.

### RAG Pipeline

```text
User Question
      ↓
Question Processing
      ↓
FAISS Retriever
      ↓
Relevant Document Chunks
      ↓
Context Formation
      ↓
Groq LLM
      ↓
Final Answer
```

This approach is particularly useful for question answering over private or domain-specific documents.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │       User           │
                         │                      │
                         │   Upload PDF         │
                         │   Ask Question       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Streamlit UI     │
                         └──────────┬───────────┘
                                    │
                         Uploaded PDF
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    PyPDFLoader       │
                         │                      │
                         │ Extract PDF Text     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ RecursiveCharacter   │
                         │ TextSplitter         │
                         │                      │
                         │ Chunk Size: 1000     │
                         │ Overlap: 150         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ HuggingFace          │
                         │ Embeddings            │
                         │                      │
                         │ all-MiniLM-L6-v2     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FAISS          │
                         │  Vector Database     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Retriever       │
                         │                      │
                         │ Top K = 4            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Prompt Template   │
                         │                      │
                         │ Context + Question   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Groq LLM        │
                         │                      │
                         │ openai/gpt-oss-20b   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    RAG Answer        │
                         └──────────────────────┘
```

---

# 🔄 Application Workflow

## Step 1 — Upload PDF

The user uploads a PDF through the Streamlit interface.

```python
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)
```

The uploaded document is temporarily saved for processing.

---

## Step 2 — Load PDF

The application uses `PyPDFLoader` to extract the document content.

```python
loader = PyPDFLoader("temp.pdf")
documents = loader.load()
```

Each page is represented as a document object.

The application also displays the number of pages loaded.

Example:

```text
📄 Pages loaded: 227
```

---

## Step 3 — Split the Document

Large documents are divided into smaller chunks using:

**RecursiveCharacterTextSplitter**

Configuration:

```text
Chunk Size   = 1000
Chunk Overlap = 150
```

The overlap helps preserve contextual information between neighboring chunks.

Example:

```text
Large PDF
    ↓
Page Text
    ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

---

## Step 4 — Generate Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings represent the semantic meaning of text as numerical vectors.

For example:

```text
"Machine learning is a subset of AI"
                ↓
        Numerical Vector
                ↓
[0.021, -0.145, 0.382, ...]
```

Semantically similar text produces vectors that are close to each other in vector space.

---

## Step 5 — Create FAISS Vector Database

The generated embeddings are stored in a **FAISS vector database**.

FAISS is used for efficient similarity search.

```python
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)
```

The vector database allows the application to quickly identify chunks that are most relevant to the user's question.

---

## Step 6 — Create Retriever

The FAISS database is converted into a retriever.

The application retrieves the top **4 relevant chunks**:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)
```

For example:

```text
User Question
      ↓
"Explain supervised learning"
      ↓
FAISS similarity search
      ↓
Top 4 relevant chunks
      ↓
Retrieved context
```

---

## Step 7 — Send Context to the LLM

The retrieved document chunks are formatted into context.

The application uses a LangChain prompt template.

The prompt instructs the LLM to:

* Use only the provided context
* Answer the user's question
* Avoid outside knowledge
* Say:

```text
I don't know based on the provided PDF.
```

when the answer is not available in the retrieved context.

This helps keep the chatbot focused on the uploaded document.

---

## Step 8 — Generate Final Answer

The retrieved context is sent to the Groq LLM.

Current model configured in the application:

```text
openai/gpt-oss-20b
```

The model generates the final natural-language answer.

The answer is then displayed in the Streamlit interface.

---

# 🛠️ Tech Stack

## Programming Language

### Python

Used for:

* Application development
* PDF processing
* Text processing
* Embedding generation
* Vector search
* LLM integration
* RAG pipeline implementation

---

## Frontend / User Interface

### Streamlit

Used to build the interactive web interface.

Features used:

* PDF uploader
* Text input
* Status messages
* Loading spinner
* Application layout

---

## RAG Framework

### LangChain

Used for building the Retrieval-Augmented Generation pipeline.

Main LangChain components used:

```text
LangChain Community
LangChain Core
LangChain Text Splitters
LangChain HuggingFace
LangChain Groq
```

---

## PDF Processing

### PyPDFLoader

Used to load and extract text from PDF documents.

---

## Text Splitting

### RecursiveCharacterTextSplitter

Used to divide large documents into manageable chunks.

Configuration:

```text
chunk_size = 1000
chunk_overlap = 150
```

---

## Embedding Model

### HuggingFace Sentence Transformers

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used to convert text chunks into vector representations.

---

## Vector Database

### FAISS

FAISS is used for similarity search over the generated embeddings.

---

## Large Language Model

### Groq

Groq provides the LLM inference layer.

Current model:

```text
openai/gpt-oss-20b
```

---

## Environment Management

### python-dotenv

Used to load environment variables during local development.

The Groq API key is stored securely and is not hard-coded into the source code.

---

## Deployment

### Streamlit Community Cloud

The application is deployed as a publicly accessible Streamlit application.

---

# 📦 Project Dependencies

The project uses the following major Python packages:

```text
streamlit
python-dotenv
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-groq
huggingface-hub
sentence-transformers
faiss-cpu
pypdf
```

---

# 📁 Project Structure

```text
PDF-RAG-ChatBot/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── data/
│   └── ...
│
├── .env
│   └── Local API key
│
└── venv/
    └── Local Python environment
```

> `.env` and `venv/` should never be committed to GitHub.

---

# 🔐 Environment Variables

The application requires a Groq API key.

For local development, create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

The application loads the key using:

```python
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

For Streamlit Cloud deployment, the key is configured using Streamlit Secrets.

Example:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### Security

Never:

* Commit `.env` to GitHub
* Hard-code API keys inside `app.py`
* Share your API key publicly
* Upload API keys in screenshots
* Put API keys inside README files

---

# 🚀 Local Installation

## 1. Clone the Repository

Clone the project repository to your local machine.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd PDF-RAG-ChatBot
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 5. Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💻 How to Use

### Step 1

Open the application.

### Step 2

Click:

```text
Upload your PDF
```

### Step 3

Select a PDF document.

### Step 4

Wait for the document processing.

The application will display:

```text
📄 Pages loaded
🧩 Chunks created
✅ HuggingFace Embeddings created
✅ FAISS vector database created
✅ Retriever created
✅ Groq LLM created
```

### Step 5

Enter a question.

Example:

```text
What is Machine Learning?
```

### Step 6

The RAG pipeline retrieves relevant information and generates the answer.

---

# 🧪 Example

### Input PDF

A Machine Learning study notes PDF.

### Question

```text
What is Machine Learning?
```

### RAG Process

```text
Question
   ↓
FAISS similarity search
   ↓
Top 4 relevant chunks
   ↓
Retrieved context
   ↓
Prompt Template
   ↓
Groq LLM
   ↓
Generated Answer
```

### Example Answer

```text
Machine Learning (ML) is a branch of Artificial Intelligence (AI)
that enables computers to learn from data and improve their
performance without being explicitly programmed.
```

The answer is generated using the retrieved content from the uploaded document.

---

# 📊 Current RAG Configuration

| Component           | Configuration                  |
| ------------------- | ------------------------------ |
| PDF Loader          | PyPDFLoader                    |
| Chunking            | RecursiveCharacterTextSplitter |
| Chunk Size          | 1000                           |
| Chunk Overlap       | 150                            |
| Embedding Model     | all-MiniLM-L6-v2               |
| Vector Database     | FAISS                          |
| Retrieved Documents | Top 4                          |
| LLM Provider        | Groq                           |
| LLM Model           | openai/gpt-oss-20b             |
| Temperature         | 0                              |
| UI Framework        | Streamlit                      |
| Deployment          | Streamlit Community Cloud      |

---

# 🎯 Key Learning Outcomes

This project helped implement and understand:

* Retrieval-Augmented Generation
* Large Language Models
* Vector embeddings
* Semantic search
* Vector databases
* Document loaders
* Text chunking
* Similarity retrieval
* Prompt engineering
* LangChain pipelines
* Groq API integration
* HuggingFace embeddings
* FAISS
* Streamlit application development
* Environment variables
* API key security
* Cloud deployment
* Git and GitHub

---

# 👨‍💻 Author

**Vikrant Jadhav**

Computer Science Graduate | AI/ML & Generative AI Enthusiast

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* Retrieval-Augmented Generation
* Large Language Models
* Agentic AI
* Data Science
* Full Stack Development

---

# ⭐ If You Like This Project

If you find this project useful or interesting:

* ⭐ Star the repository
* 🍴 Fork the project
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute to the project

---

# 📜 License

This project is created for educational, portfolio, and learning purposes.

You may modify and improve the project according to your requirements.

# 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployment process is:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements.txt
       ↓
Configure GROQ_API_KEY
       ↓
Run app.py
       ↓
Public Web Application
```

---

# ☁️ Streamlit Cloud Deployment

## Step 1 — Push Project to GitHub

Make sure the following files are committed:

```text
app.py
requirements.txt
README.md
.gitignore
```

Do not commit:

```text
.env
venv/
.venv/
__pycache__/
```

---

## Step 2 — Connect GitHub Repository

Create a Streamlit Cloud application and select:

```text
Repository:
vikrantjadhav09/PDF-RAG-ChatBot

Branch:
main

Main file:
app.py
```

---

## Step 3 — Configure Secrets

Add the Groq API key to Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The API key should never be stored directly in the GitHub repository.

---

## Step 4 — Deploy

After configuration, deploy the application.

Streamlit Cloud installs the dependencies from:

```text
requirements.txt
```

and executes:

```text
app.py
```

---

# 🔄 End-to-End RAG Pipeline

The complete application pipeline can be summarized as:

```text
                         USER
                          │
                          ▼
                  Upload PDF Document
                          │
                          ▼
                    PyPDFLoader
                          │
                          ▼
                    Extract Text
                          │
                          ▼
              Recursive Text Splitter
                          │
                          ▼
                    Text Chunks
                          │
                          ▼
              HuggingFace Embeddings
                          │
                          ▼
                   Vector Embeddings
                          │
                          ▼
                       FAISS
                 Vector Database
                          │
                          │
                          │
USER QUESTION ────────────┤
                          ▼
                     Retriever
                          │
                    Top 4 Chunks
                          │
                          ▼
                  Retrieved Context
                          │
                          ▼
                  Prompt Template
                          │
                          ▼
                    Groq LLM
                          │
                          ▼
                   Generated Answer
                          │
                          ▼
                     Streamlit UI
                          │
                          ▼
                         USER
```

---

# 🧩 Why Each Component Is Used

## PyPDFLoader

Used to extract text and page-level information from PDF documents.

---

## RecursiveCharacterTextSplitter

Large documents cannot always be efficiently processed as one large block.

The splitter divides the content into smaller chunks.

```text
Large Document
      ↓
Small Chunks
      ↓
Embeddings
      ↓
Vector Database
```

The configured overlap helps maintain continuity between neighboring chunks.

---

## HuggingFace Embeddings

Text cannot be directly compared efficiently based on meaning.

Embeddings convert text into numerical vectors.

For example:

```text
"Machine learning uses data"
             ↓
        Embedding Vector
```

The vector representation allows semantic similarity search.

---

## FAISS

FAISS stores the vectors and performs similarity search.

For example:

```text
Question:
"What is supervised learning?"

              ↓

FAISS similarity search

              ↓

Relevant chunks from PDF
```

Only the most relevant chunks are passed to the next stage.

---

## Retriever

The retriever is responsible for finding the most relevant document chunks.

Current configuration:

```python
search_kwargs={
    "k": 4
}
```

Therefore, the application retrieves up to four relevant chunks for each question.

---

## Prompt Template

The prompt provides instructions to the LLM.

The main instruction is:

```text
Answer the question using ONLY the provided context.
```

This helps reduce answers based on information outside the uploaded document.

If the answer is not available in the retrieved context, the application instructs the model to respond:

```text
I don't know based on the provided PDF.
```

---

## Groq LLM

The retrieved context is passed to the Groq-hosted model.

Current model:

```text
openai/gpt-oss-20b
```

The model generates the final natural-language response.

---

# 🧠 RAG vs Traditional Chatbot

A traditional chatbot may answer using the knowledge available within its underlying model.

This project follows a different approach.

```text
Traditional Chatbot

User Question
      ↓
LLM
      ↓
Answer
```

The PDF RAG chatbot:

```text
User Question
      ↓
Search PDF Knowledge
      ↓
Retrieve Relevant Context
      ↓
LLM
      ↓
Answer
```

The second approach allows the application to answer questions about documents that were not part of the application's original knowledge.

---

# 🔐 Security Considerations

The project uses an API key for Groq.

The following practices should be followed:

### Never commit API keys

Bad:

```python
GROQ_API_KEY = "gsk_xxxxxxxxx"
```

Good:

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

For Streamlit Cloud, use Streamlit Secrets.

---

# ⚠️ Current Limitations

This is a working portfolio-level RAG application, but it has several areas that can be improved.

## 1. Single PDF Processing

The current interface is designed around processing the uploaded PDF during the current application interaction.

A future version could support multiple documents simultaneously.

---

## 2. Temporary PDF File

The uploaded PDF is temporarily stored as:

```text
temp.pdf
```

A production-oriented implementation could use safer temporary-file handling.

---

## 3. Embedding Model Loading

The HuggingFace embedding model is initialized during document processing.

Caching the embedding model could improve performance.

---

## 4. Vector Database Persistence

The current application creates a FAISS index from the uploaded document.

A more advanced version could persist vector indexes and reuse them.

---

## 5. No Conversation Memory

The current application focuses on individual PDF questions.

A future version could maintain conversation history.

For example:

```text
User:
What is Machine Learning?

Assistant:
Machine Learning is...

User:
What are its types?

Assistant:
The main types are...
```

---

## 6. No Source Citation UI

The current answer displays the generated response but does not show the exact PDF pages used to generate the answer.

A future version could display:

```text
Answer
   ↓
Sources
   ├── Page 12
   ├── Page 15
   └── Page 16
```

This would improve transparency and trust.

---

## 7. Large PDF Performance

Very large PDFs may require optimization.

Possible improvements include:

* Better chunking strategies
* Batch embedding
* Persistent vector databases
* Caching
* Asynchronous processing
* Background document processing

---

# 🚀 Future Improvements

The project can be extended into a more advanced production-style RAG system.

## 🔹 Multiple PDF Support

Allow users to upload multiple PDFs:

```text
PDF 1
PDF 2
PDF 3
PDF 4
   ↓
Combined Knowledge Base
   ↓
Retriever
   ↓
LLM
```

---

## 🔹 Conversational RAG

Add chat history so users can ask follow-up questions.

Example:

```text
User:
What is regression?

Assistant:
Regression is...

User:
What are its types?

Assistant:
The main types of regression are...
```

---

## 🔹 Source Citations

Display the exact page numbers used for generating the answer.

Example:

```text
Answer:
...

Sources:
📄 Page 24
📄 Page 27
📄 Page 31
```

---

## 🔹 Better RAG Evaluation

Evaluate the retrieval and generation quality using metrics such as:

* Context Precision
* Context Recall
* Faithfulness
* Answer Relevancy

RAG evaluation frameworks can be integrated in a future version.

---

## 🔹 Persistent Vector Database

Instead of creating the vector database every time, the application could use:

```text
ChromaDB
Pinecone
Milvus
Qdrant
Weaviate
```

depending on the application's requirements.

---

## 🔹 Hybrid Search

A future version could combine:

```text
Semantic Search
       +
Keyword Search
       ↓
Hybrid Retrieval
```

This can improve retrieval for technical terms, names, identifiers, and exact keywords.

---

## 🔹 Reranking

A reranking model could be added after initial retrieval:

```text
Question
   ↓
FAISS
   ↓
Top 10 chunks
   ↓
Reranker
   ↓
Best 4 chunks
   ↓
LLM
```

This can improve the relevance of the final context.

---

## 🔹 Authentication

The application could eventually support:

* User accounts
* Login
* Authentication
* User-specific documents
* Document management

---

## 🔹 Cloud Storage

Uploaded PDFs could be stored in services such as:

```text
AWS S3
Firebase Storage
Cloudinary
Google Cloud Storage
```

instead of temporary local storage.

---

# 📈 Possible Production Architecture

A more production-oriented version could look like:

```text
                       USERS
                         │
                         ▼
                 ┌───────────────┐
                 │   Frontend    │
                 │ React / Next  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    FastAPI    │
                 │    Backend    │
                 └───────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      PDF Storage    Vector DB      LLM API
          │              │              │
          ▼              ▼              ▼
        PDFs          Embeddings       Groq
                         │
                         ▼
                    Retriever
                         │
                         ▼
                    RAG Pipeline
                         │
                         ▼
                      Answer
```

---

# 🧪 Testing Scenarios

The application can be tested using different types of questions.

## Direct Questions

```text
What is Machine Learning?
```

---

## Conceptual Questions

```text
Explain supervised learning.
```

---

## Comparison Questions

```text
What is the difference between supervised and unsupervised learning?
```

---

## Definition Questions

```text
What is overfitting?
```

---

## Document-Grounded Questions

```text
According to the PDF, what are the advantages of decision trees?
```

---

## Out-of-Context Questions

```text
What is the population of India?
```

The application is instructed to respond:

```text
I don't know based on the provided PDF.
```

when the requested information is not available in the retrieved context.

---

# 💡 Example Use Cases

This type of RAG chatbot can be useful for:

### 🎓 Education

Students can upload:

* Lecture notes
* Study material
* Textbooks
* Research papers

and ask questions about them.

---

### 🏢 Business

Employees can upload:

* Company policies
* Internal documents
* Reports
* Manuals

and retrieve information using natural language.

---

### ⚖️ Legal Documents

Users could interact with:

* Contracts
* Agreements
* Legal documents

subject to appropriate privacy and legal safeguards.

---

### 🔬 Research

Researchers could use RAG for:

* Research papers
* Technical documentation
* Academic notes
* Literature review support

---

### 📚 Personal Knowledge Base

Users can upload personal documents and interact with them through natural-language questions.

---

# 📌 Important Technical Concepts Demonstrated

This project demonstrates practical implementation of:

```text
Python
   ↓
Document Processing
   ↓
Text Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Semantic Search
   ↓
Information Retrieval
   ↓
Prompt Engineering
   ↓
Large Language Model
   ↓
RAG
   ↓
Streamlit Deployment
```

---

# 💼 Resume Project Description

You can describe this project on your resume as:

**PDF RAG Chatbot | Python, LangChain, HuggingFace, FAISS, Groq, Streamlit**

* Built an AI-powered PDF question-answering application using Retrieval-Augmented Generation (RAG), enabling users to query uploaded documents using natural language.
* Implemented PDF extraction, recursive text chunking, HuggingFace embeddings, FAISS similarity search, and top-4 contextual retrieval using LangChain.
* Integrated Groq's `openai/gpt-oss-20b` model with a context-grounded prompt to generate document-based responses and deployed the application using Streamlit Community Cloud.

---

# 🗣️ Interview Explanation

If an interviewer asks:

### "Explain your PDF RAG Chatbot project."

A simple explanation is:

> I built a PDF-based RAG chatbot using Python, LangChain, HuggingFace embeddings, FAISS, Groq, and Streamlit. The user uploads a PDF, which is loaded using PyPDFLoader and split into smaller chunks using RecursiveCharacterTextSplitter. I convert those chunks into embeddings using the all-MiniLM-L6-v2 model and store them in FAISS. When the user asks a question, the retriever performs similarity search and retrieves the top four relevant chunks. Those chunks are passed as context to a Groq LLM through a LangChain prompt, and the model generates an answer based on the retrieved PDF content. I deployed the application using Streamlit Community Cloud.

---

# ⭐ Project Highlights

```text
✅ End-to-end RAG implementation
✅ PDF document processing
✅ Semantic vector search
✅ HuggingFace embeddings
✅ FAISS vector database
✅ LangChain pipeline
✅ Groq LLM integration
✅ Context-grounded responses
✅ Secure API key handling
✅ Streamlit UI
✅ Cloud deployment
```

---

# 📊 Project Architecture Summary

```text
             PDF DOCUMENT
                   │
                   ▼
             PyPDFLoader
                   │
                   ▼
              Text Pages
                   │
                   ▼
        Recursive Text Splitter
                   │
                   ▼
              Text Chunks
                   │
                   ▼
       HuggingFace Embeddings
                   │
                   ▼
             FAISS Index
                   │
                   │
                   │
USER QUESTION ─────┘
                   │
                   ▼
              Retriever
                   │
                   ▼
          Top 4 Relevant Chunks
                   │
                   ▼
             Prompt Template
                   │
                   ▼
               Groq LLM
                   │
                   ▼
             Final Answer
```

---

# 🔗 Project Links

### GitHub Repository

**PDF-RAG-ChatBot**

### Live Application

**PDF RAG Chatbot — Streamlit**

---

# 🙌 Acknowledgements

This project was developed as part of hands-on learning in:

* Python
* Machine Learning
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* LangChain
* Vector Databases
* Streamlit

---

# 📬 Contact

**Vikrant Jadhav**

If you are interested in discussing:

* AI/ML
* Generative AI
* RAG
* LLM applications
* Data Science
* Full Stack AI applications

feel free to connect.

---

# ⭐ Final Project Summary

**PDF RAG Chatbot** is an end-to-end Generative AI application that demonstrates how modern RAG systems can transform unstructured PDF documents into an interactive question-answering system.

The project combines:

```text
PDF Processing
      +
Text Chunking
      +
Embeddings
      +
Vector Search
      +
Information Retrieval
      +
Prompt Engineering
      +
Large Language Model
      +
Streamlit
      +
Cloud Deployment
```

into a single working application.

---

## 🚀 Built with

```text
🐍 Python
🦜 LangChain
🤗 HuggingFace
⚡ FAISS
🤖 Groq
🎈 Streamlit
📄 PyPDF
🔎 RAG
```

---

**Built with Python + LangChain + HuggingFace + FAISS + Groq + Streamlit ❤️**
