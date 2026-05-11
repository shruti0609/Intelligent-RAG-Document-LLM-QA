from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader("document loaders/deeplearning.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Embeddings (FREE)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Create FAISS vector store
vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

# ✅ Save FAISS DB locally
vectorstore.save_local("faiss_db")

print("DB created successfully ✅")