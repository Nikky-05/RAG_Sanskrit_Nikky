import os
from config import SANSKRIT_FILE, DATA_DIR


def load_text_file(filepath):
    """read a single txt file and return its content"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def load_all_documents():
    """load all txt files from data folder"""
    documents = []

    # check if main sanskrit file exists
    if os.path.exists(SANSKRIT_FILE):
        content = load_text_file(SANSKRIT_FILE)
        documents.append({"source": SANSKRIT_FILE, "text": content})
        print(f"loaded: {SANSKRIT_FILE}")
    else:
        print(f"file not found: {SANSKRIT_FILE}")

    # also pick up any other txt files in data folder
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".txt") and fpath != SANSKRIT_FILE:
            content = load_text_file(fpath)
            documents.append({"source": fpath, "text": content})
            print(f"loaded: {fpath}")

    print(f"total documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents()
    for d in docs:
        print(f"\n--- {d['source']} ---")
        print(d["text"][:200] + "...")
