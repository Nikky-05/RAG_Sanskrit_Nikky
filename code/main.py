import os
import sys
import time
import psutil

# fix unicode output on windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# add code directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FAISS_INDEX_PATH, LLM_MODEL_PATH
from build_index import build_pipeline, load_index
from query_rag import RAGPipeline


def get_memory_mb():
    """get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_model_exists():
    """check if LLM model file is downloaded"""
    if not os.path.exists(LLM_MODEL_PATH):
        print("=" * 50)
        print("LLM model not found!")
        print(f"expected path: {LLM_MODEL_PATH}")
        print()
        print("download the model file:")
        print("  model: TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_K_M)")
        print("  from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print("  file: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        print(f"  put it in: {os.path.dirname(LLM_MODEL_PATH)}")
        print("=" * 50)
        return False
    return True


def main():
    print("Sanskrit RAG System")
    print("=" * 50)

    mem_start = get_memory_mb()
    print(f"initial memory usage: {mem_start:.1f} MB")

    # check if model exists
    if not check_model_exists():
        sys.exit(1)

    # check if index exists, build if not
    if not os.path.exists(FAISS_INDEX_PATH):
        print("\nFAISS index not found, building now...")
        t0 = time.time()
        build_pipeline()
        print(f"index build time: {time.time() - t0:.2f}s")
    else:
        print("FAISS index found, loading...")

    # initialize RAG pipeline
    print("\ninitializing RAG pipeline...")
    t0 = time.time()
    rag = RAGPipeline()
    load_time = time.time() - t0
    mem_after_load = get_memory_mb()
    print(f"pipeline load time: {load_time:.2f}s")
    print(f"memory after loading: {mem_after_load:.1f} MB")

    # interactive query loop
    print("\n" + "=" * 50)
    print("ready! type your question in Sanskrit or transliterated text")
    print("type 'exit' or 'quit' to stop")
    print("=" * 50)

    while True:
        print()
        query = input("your question: ").strip()

        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("bye!")
            break

        mem_before = get_memory_mb()
        start = time.time()
        answer, results = rag.query(query)
        elapsed = time.time() - start
        mem_after = get_memory_mb()

        print(f"\nanswer: {answer}")
        print(f"\n--- performance ---")
        print(f"  total latency   : {elapsed:.2f}s")
        print(f"  memory usage    : {mem_after:.1f} MB")
        print(f"  chunks retrieved: {len(results)}")
        print(f"  best score (L2) : {results[0]['score']:.4f}" if results else "")


if __name__ == "__main__":
    main()
