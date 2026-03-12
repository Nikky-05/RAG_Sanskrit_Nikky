import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text):
    """basic cleaning for sanskrit text"""
    # remove multiple spaces
    text = re.sub(r" +", " ", text)
    # remove multiple newlines but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip leading/trailing whitespace
    text = text.strip()
    return text


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """split text into overlapping chunks
    tries to break at sentence boundaries (। or ॥ or newline)
    """
    text = clean_text(text)
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            # last chunk, take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # try to find a good break point near the end
        # look for sanskrit sentence enders: । ॥ or newline
        segment = text[start:end]
        best_break = -1

        for sep in ["॥", "।", "\n"]:
            pos = segment.rfind(sep)
            if pos > chunk_size // 2:  # dont break too early
                best_break = max(best_break, pos + len(sep))

        if best_break > 0:
            chunk = segment[:best_break].strip()
            next_start = start + best_break - overlap
        else:
            # no good break point found, just cut at chunk_size
            chunk = segment.strip()
            next_start = start + chunk_size - overlap

        if chunk:
            chunks.append(chunk)
        start = next_start

    return chunks


def preprocess_documents(documents):
    """take list of document dicts, return list of chunks with metadata"""
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        chunks = split_into_chunks(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": i
            })

    print(f"total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # quick test
    sample = "यह पहला वाक्य है। यह दूसरा वाक्य है। " * 20
    chunks = split_into_chunks(sample, chunk_size=100, overlap=20)
    print(f"got {len(chunks)} chunks from sample text")
    for i, c in enumerate(chunks):
        print(f"chunk {i}: {c[:60]}...")
