import time
import numpy as np
from rank_bm25 import BM25Okapi
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL_PATH,
    TOP_K,
    MAX_TOKENS,
    TEMPERATURE,
    CONTEXT_LENGTH,
)
from build_index import load_index


def tokenize_sanskrit(text):
    """tokenizer for Sanskrit/Devanagari text with prefix expansion for morphological variants"""
    import re
    tokens = re.split(r"[\s।॥,;:\-\(\)\[\]\"\']+", text)
    tokens = [t for t in tokens if len(t) > 1]

    # add character prefixes (min 3 chars) to handle Sanskrit vibhakti/sandhi variants
    # e.g. "भोजराजा" and "भोजराज्ञा" both share prefix "भोजराज"
    expanded = list(tokens)
    for t in tokens:
        if len(t) >= 5:
            expanded.append(t[:len(t) - 1])
        if len(t) >= 7:
            expanded.append(t[:len(t) - 2])
    return expanded


class VectorRetriever:
    """FAISS-based dense vector retrieval"""

    def __init__(self, index, chunks, embed_model):
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model

    def search(self, query, top_k=TOP_K):
        query_vec = self.embed_model.encode([query], device="cpu")
        query_vec = np.array(query_vec, dtype="float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = {}
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results[int(idx)] = float(distances[0][i])
        return results


class BM25Retriever:
    """BM25 keyword-based sparse retrieval"""

    def __init__(self, chunks):
        self.chunks = chunks
        corpus = [tokenize_sanskrit(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=TOP_K):
        query_tokens = tokenize_sanskrit(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = {}
        for idx in top_indices:
            if scores[idx] > 0:
                results[int(idx)] = float(scores[idx])
        return results


class HybridRetriever:
    """combines FAISS vector search + BM25 keyword search using reciprocal rank fusion"""

    def __init__(self, index, chunks, embed_model):
        self.chunks = chunks
        self.vector = VectorRetriever(index, chunks, embed_model)
        self.bm25 = BM25Retriever(chunks)

    def search(self, query, top_k=TOP_K):
        # get more candidates from each retriever for better fusion
        n_candidates = min(top_k * 3, len(self.chunks))

        vector_results = self.vector.search(query, top_k=n_candidates)
        bm25_results = self.bm25.search(query, top_k=n_candidates)

        # reciprocal rank fusion (RRF) with k=60
        # BM25 gets 2x weight because Sanskrit keyword matching handles vibhakti better
        rrf_k = 60
        bm25_weight = 2.0
        fused_scores = {}

        # rank vector results (lower L2 distance = better, so sort ascending)
        vector_sorted = sorted(vector_results.items(), key=lambda x: x[1])
        for rank, (idx, _) in enumerate(vector_sorted):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        # rank BM25 results (higher score = better, so sort descending)
        bm25_sorted = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
        for rank, (idx, _) in enumerate(bm25_sorted):
            fused_scores[idx] = fused_scores.get(idx, 0) + bm25_weight / (rrf_k + rank + 1)

        # sort by fused score (higher = better)
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            vec_score = vector_results.get(idx, -1)
            bm25_score = bm25_results.get(idx, 0)
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": round(score, 4),
                "vector_score": round(vec_score, 4) if vec_score >= 0 else None,
                "bm25_score": round(bm25_score, 4) if bm25_score > 0 else None,
            })

        return results


class Generator:
    def __init__(self, model_path=LLM_MODEL_PATH):
        print("loading LLM model (this may take a minute on CPU)...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_LENGTH,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("LLM loaded successfully")

    def generate(self, query, context_chunks):
        """generate answer using retrieved context"""
        context = "\n---\n".join([c["text"] for c in context_chunks])

        prompt = f"""Below is some context from Sanskrit documents, followed by a question.
Use the context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        output = self.llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            repeat_penalty=1.3,
            stop=["Question:", "\n\n\n"],
        )

        answer = output["choices"][0]["text"].strip()
        return answer


class RAGPipeline:
    def __init__(self):
        # load index
        index, chunks = load_index()
        if index is None:
            raise Exception("index not found. run build_index.py first")

        # load embedding model (forced CPU)
        print("loading embedding model (CPU)...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        # setup hybrid retriever and generator
        print("building BM25 index for hybrid retrieval...")
        self.retriever = HybridRetriever(index, chunks, embed_model)
        self.generator = Generator()

    def query(self, question):
        """full RAG: hybrid retrieve then generate answer"""
        # retrieve
        t0 = time.time()
        results = self.retriever.search(question, top_k=TOP_K)
        retrieval_time = time.time() - t0

        print(f"\nretrieved {len(results)} chunks via hybrid search (retrieval: {retrieval_time:.3f}s):")
        for i, r in enumerate(results):
            vec = f"L2={r['vector_score']}" if r.get('vector_score') is not None else "no-vec"
            bm25 = f"BM25={r['bm25_score']}" if r.get('bm25_score') is not None else "no-bm25"
            print(f"  [{i+1}] fused={r['score']:.4f} ({vec}, {bm25}) | {r['text'][:70]}...")

        # generate
        t0 = time.time()
        answer = self.generator.generate(question, results)
        generation_time = time.time() - t0
        print(f"generation time: {generation_time:.2f}s")

        return answer, results


if __name__ == "__main__":
    rag = RAGPipeline()
    q = "कालीदासः कः आसीत्?"
    print(f"\nquery: {q}")
    answer, _ = rag.query(q)
    print(f"\nanswer: {answer}")
