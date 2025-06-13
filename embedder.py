import os
import pickle
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index: faiss.Index, metadata: list[dict]):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> list[dict]:
        # Perform search, returning metadata and distance scores
        D, I = self.index.search(query_embedding, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            meta = self.metadata[idx].copy()
            meta['score'] = float(dist)
            results.append(meta)
        return results

embed_model_id = 'intfloat/e5-small-v2'
model_kwargs = {"device": "cpu", "trust_remote_code": True}
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

# Paths for persisting index and metadata
INDEX_PATH = 'faiss.index'
META_PATH = 'faiss_meta.pkl'


def batch_embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Embed texts in batches to improve CPU throughput.
    Uses bulk embed_documents if available, else parallel embed_query.
    Returns a float32 numpy array of embeddings.
    """
    # Bulk embedding if supported
    if hasattr(embeddings, 'embed_documents'):
        emb_list = embeddings.embed_documents(texts)
    else:
        # Parallelize embed_query calls
        with ThreadPoolExecutor(max_workers=4) as executor:
            emb_list = list(executor.map(embeddings.embed_query, texts))
    return np.array(emb_list, dtype='float32')

def create_index(
    documents: list[dict],
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
    use_approx: bool = True
) -> FAISSIndex:
    """
    Build or rebuild the FAISS index from document chunks,
    persisting the index and metadata for fast reloads.
    """
    texts = [doc['text'] for doc in documents]
    metadata = [{'filename': doc['filename'], 'text': doc['text']} for doc in documents]

    # 1) Embed all texts in batches
    embeddings_matrix = batch_embed(texts)
    dim = embeddings_matrix.shape[1]

    # 2) Choose index type: HNSW for approximate CPU speed, or exact L2
    if use_approx:
        # HNSW index for fast approximate nearest neighbor
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
    else:
        # Exact L2 distance index
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings_matrix)

    # 3) Persist the index and metadata to disk
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    return FAISSIndex(index, metadata)

def load_index(
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH
) -> FAISSIndex:
    """
    Load an existing FAISS index and its metadata from disk.
    """
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Index or metadata file not found; run create_index first."
        )

    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    return FAISSIndex(index, metadata)


def retrieve_docs(query: str, faiss_idx: FAISSIndex, k: int = 5) -> list[dict]:
    """
    Embed the query and perform k-NN search, returning metadata with distance scores.
    """
    # Embed the query
    q_emb = batch_embed([query])[0].reshape(1, -1)
    # Search and return results
    return faiss_idx.similarity_search(q_emb, k)