import numpy as np
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


class Retriever:
    def __init__(self, index, chunks, embed_model):
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model

    def search(self, query, top_k=TOP_K):
        """find top_k most relevant chunks for the query"""
        query_vec = self.embed_model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(distances[0][i]),
                })
        return results


class Generator:
    def __init__(self, model_path=LLM_MODEL_PATH):
        print("loading LLM model (this may take a minute on CPU)...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_LENGTH,
            n_threads=4,
            verbose=False,
        )
        print("LLM loaded successfully")

    def generate(self, query, context_chunks):
        """generate answer using retrieved context"""
        # build context string from retrieved chunks
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

        # load embedding model
        print("loading embedding model...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # setup retriever and generator
        self.retriever = Retriever(index, chunks, embed_model)
        self.generator = Generator()

    def query(self, question):
        """full RAG: retrieve relevant chunks then generate answer"""
        # retrieve
        results = self.retriever.search(question, top_k=TOP_K)

        print(f"\nretrieved {len(results)} chunks:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.4f} | {r['text'][:80]}...")

        # generate
        answer = self.generator.generate(question, results)
        return answer, results


if __name__ == "__main__":
    rag = RAGPipeline()
    q = "कालीदासः कः आसीत्?"
    print(f"\nquery: {q}")
    answer, _ = rag.query(q)
    print(f"\nanswer: {answer}")
