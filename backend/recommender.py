from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

model = None
index = None
data = None

def load_resources():
    global model, index, data

    if model is None:
        print("Loading model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if data is None:
        data = pd.read_csv("data/shl_assessments.csv")

    if index is None:
        embeddings = np.load("data/embeddings.npy")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)


def recommend(query, top_k=10):
    load_resources()

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        row = data.iloc[idx]
        results.append({
            "title": row["title"],
            "url": row["url"]
        })

    return results