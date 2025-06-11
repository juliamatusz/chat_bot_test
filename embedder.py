import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)  # D: distances, I: indices
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

# Load model once globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = SentenceTransformer("intfloat/e5-small-v2", device=device)

def create_index(documents):
    texts = [doc["text"] for doc in documents]
    metadata = [{"filename": doc["filename"], "text": doc["text"]} for doc in documents]

    # Generate embeddings
    embeddings_matrix = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    # Return a FAISSIndex object that contains both the index and metadata
    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    results = faiss_index.similarity_search(query_embedding, k=k)
    return results
