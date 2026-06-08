#Creating a rag retriever pipeline 
#To import packages 

import pickle 
from pathlib import Path 
from functools import lru_cache 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 


#Taking the paths 
KB_DIR = Path(__file__).parent.parent / "knowledge_base" / "index"
INDEX_FILE = KB_DIR / "kb.index"
CHUNKS_FILE  = KB_DIR / "chunks.pkl"

EMBED_MODEL  = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5 

#Building a class called rag retriever 

class RagRetriever:
    #For any class first you do the following


    #Add the init self method 

    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"Knowledge base index not found at {INDEX_FILE}\n"
                f"Run: python knowledge_base/build_kb.py"
            )
        
        print("[RAG] Loading knowledge base...")
        self._model = SentenceTransformer(EMBED_MODEL)
        self._index = faiss.read_index(str(INDEX_FILE))

        with open(CHUNKS_FILE, "rb") as f:
            self._chunks = pickle.load(f)

        print(f"[RAG] Loaded {self._index.ntotal} chunks")

    #Retrieving from the rag 
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        #To find the most relevant 
        #Step 1) Embed the query using same model
        #Step 2) Search the FAISS 
        #Step 3) Return top_k passages concatenated as a single string 

        query_vector = self._model.encode(
            [query],
            convert_to_numpy = True,
        ).astype("float32")


        #Searching FAISS 
        top_k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vector, top_k)


        #Collecting and returning passages 
        passages = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue 
            chunk = self._chunks[idx]
            passages.append(
                f"[Source: {chunk['source']}, rank {rank+1}\n{chunk['text']}]"
            )


        #If not present in passages 
        if not passages:
            return "No relevant passage "
        return "\n\n".join(passages)
    

    #Now retrieving the features
    def retrieve_features(self, feature_type: str, iso_grade: str, nominal_mm: float = None) -> str:
        parts = [feature_type.replace("_", ""), iso_grade, "machining process"]
        if nominal_mm: 
            parts.append(f"{nominal_mm}mm")
        
        query = "".join(parts)
        return self.retrieve(query)

@lru_cache(maxsize=1)
def get_retriever() -> RagRetriever:
    return RagRetriever()


    