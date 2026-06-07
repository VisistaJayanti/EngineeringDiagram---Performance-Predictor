#So this is the file needed to build the RAG pipeline 


#First importing packages for the same 
import os 
import io 
import json
import pickle
from pathlib import Path 
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np 


#Now the paths needed 
KB_DIR = Path(__file__).parent
DOCS_DIR = KB_DIR / "documents"
INDEX_DIR = KB_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40 
EMBED_MODEL = "all-MiniLM-L6-v2"


#Now taking the documents to load 
def load_documents() -> list[dict]:
    #Creating empty list first 
    docs = []

    #Now parsing through the fiels 
    #First sorting 
   

    #Now iterating through the text file 
    for txt_file in sorted(DOCS_DIR.glob("*txt")):
        #Now content will be stripping the text file 
        content = txt_file.read_text(encoding="utf-8").strip()
        
        #Now adding to docs 
        docs.append({
            "filename" : txt_file.name,
            "content" : content,
        })
        print(f" Loaded: {txt_file.name} ({len(content)} chars)")
    return docs 

#Now chunking the document 
def chunk_documents(doc: dict) -> list[dict]:
    #Overlapping ensures the full context is capthured in one chunk 

    #To take words
    words = doc["content"].split()
    chunks = []
    start = 0
    idx = 0 


    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = "".join(words[start:end])

        chunks.append({
            "chunk_id" : f"{doc['filename']}__chunk_{idx}",
            "text" : chunk_text,
            "source" : doc["filename"],
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP 
        idx += 1 
    return chunks 


#Building the index
def build_index(chunks: list[dict], model: SentenceTransformer):
    print(f"\nEmbedding {len(chunks)} chunks...")
    text = [c["text"] for c in chunks]
    embeddings = model.encode(
        text,
        show_progress_bar = True,
        batch_size = 32,
        convert_to_numpy = True,
    )
    embeddings = embeddings.astype("float32")

    #Building the FAISS index 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f" Index built: {index.ntotal} vectors, dimension{dim}")
    return index, embeddings 


def save(index, chunks: list[dict], embeddings):
    """Save FAISS index and chunk metadata to disk."""
    faiss.write_index(index, str(INDEX_DIR / "kb.index"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved to {INDEX_DIR}/")
    print(f"  kb.index  — FAISS vector index")
    print(f"  chunks.pkl — chunk text and metadata")

def main():
    print("=== Building knowledge base ===\n")

    print("Loading documents...")
    docs   = load_documents()
    if not docs:
        print("No .txt files found in knowledge_base/documents/")
        return

    print(f"\nChunking {len(docs)} documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_documents(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    index, embeddings = build_index(all_chunks, model)
    save(index, all_chunks, embeddings)

    print("\n=== Knowledge base ready ===")
    print(f"Run: python knowledge_base/build_kb.py  (only needed once)")

if __name__ == "__main__":
    main()
    