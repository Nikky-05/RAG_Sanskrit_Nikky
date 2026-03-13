import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    VECTORSTORE_DIR,
)
from load_documents import load_all_documents
from preprocess import preprocess_documents


def create_embeddings(chunks, model):
    """generate embeddings for all chunks using sentence-transformers on CPU"""
    texts = [c["text"] for c in chunks]
    print(f"generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16, device="cpu")
    embeddings = np.array(embeddings, dtype="float32")
    return embeddings


def build_faiss_index(embeddings):
    """build a FAISS index from embeddings"""
    dim = embeddings.shape[1]
    # using L2 distance index - simple and works well for small datasets
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors, dimension={dim}")
    return index


def save_index(index, chunks):
    """save FAISS index and chunks to disk"""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"index saved to {FAISS_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"chunks saved to {CHUNKS_PATH}")


def load_index():
    """load FAISS index and chunks from disk"""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"loaded index with {index.ntotal} vectors")
    return index, chunks


def build_pipeline():
    """full pipeline: load docs -> preprocess -> embed -> index -> save"""
    print("=" * 50)
    print("building index pipeline")
    print("=" * 50)

    # step 1: load documents
    print("\n[step 1] loading documents...")
    docs = load_all_documents()
    if not docs:
        print("no documents found. put txt files in data/ folder")
        return None, None, None

    # step 2: preprocess and chunk
    print("\n[step 2] preprocessing and chunking...")
    chunks = preprocess_documents(docs)

    # step 3: load embedding model (forced CPU)
    print("\n[step 3] loading embedding model (CPU)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    # step 4: create embeddings
    print("\n[step 4] creating embeddings...")
    embeddings = create_embeddings(chunks, model)

    # step 5: build FAISS index
    print("\n[step 5] building FAISS index...")
    index = build_faiss_index(embeddings)

    # step 6: save everything
    print("\n[step 6] saving index and chunks...")
    save_index(index, chunks)

    print("\npipeline complete!")
    return index, chunks, model


if __name__ == "__main__":
    build_pipeline()
