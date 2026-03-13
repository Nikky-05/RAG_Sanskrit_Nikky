import os

# force CPU-only: disable all GPU/CUDA access
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data folder path
DATA_DIR = os.path.join(BASE_DIR, "data")
SANSKRIT_FILE = os.path.join(DATA_DIR, "sanskrit_docs.txt")

# vectorstore
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

# model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, ".cache", "huggingface", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# embedding model - ye sentence-transformers se download hoga automatically
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chunking settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# retrieval settings
TOP_K = 3

# llm settings
MAX_TOKENS = 512
TEMPERATURE = 0.3
CONTEXT_LENGTH = 2048
