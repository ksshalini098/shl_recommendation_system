import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "models", "faiss_index.index")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    df = pickle.load(f)


def recommend(query, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "name": row["name"],
            "url": row["url"],
        })

    return results