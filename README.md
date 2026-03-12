# Sanskrit Document RAG System

A Retrieval-Augmented Generation system for querying Sanskrit documents. Runs entirely on CPU.

## Project Structure

```
RAG_Sanskrit_Nikky/
├── code/
│   ├── main.py              # entry point
│   ├── load_documents.py    # loads txt files from data/
│   ├── preprocess.py        # text cleaning and chunking
│   ├── build_index.py       # creates embeddings and FAISS index
│   ├── query_rag.py         # retriever + LLM generator
│   └── config.py            # all configuration and paths
├── data/
│   └── sanskrit_docs.txt    # source sanskrit documents
├── vectorstore/             # FAISS index (auto-generated)
├── models/                  # LLM model file goes here
├── report/
│   └── RAG_Report.pdf
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the LLM model

Download the TinyLlama GGUF model file:

- Go to: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- Download: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (~670 MB)
- Place it in the `models/` folder

### 3. Run the system

```bash
cd code
python main.py
```

On first run it will:
1. Load and preprocess the Sanskrit documents
2. Generate embeddings and build the FAISS index
3. Start an interactive query interface

After that, type any question in Sanskrit or transliterated text and get answers.

## How it works

1. **Document Loading** - Reads Sanskrit text files from `data/` folder
2. **Preprocessing** - Cleans text, splits into overlapping chunks (~400 chars)
3. **Indexing** - Generates vector embeddings using sentence-transformers (all-MiniLM-L6-v2), stores in FAISS
4. **Retrieval** - Converts user query to embedding, finds top-3 similar chunks via FAISS
5. **Generation** - Sends retrieved context + query to TinyLlama LLM, generates answer

## Technical Details

- **Embedding Model**: all-MiniLM-L6-v2 (384-dim, multilingual)
- **Vector Store**: FAISS (IndexFlatL2)
- **LLM**: TinyLlama-1.1B-Chat (Q4_K_M quantized GGUF)
- **Inference**: CPU only, no GPU required

## Example Query

```
your question: कालीदासः कः आसीत्?
```

The system retrieves relevant chunks about Kalidasa from the indexed documents and generates an answer.
