import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import pickle

DATA_PATH = "data/assessments.csv"
INDEX_PATH = "models/faiss_index.index"
META_PATH = "models/metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    df["combined_text"] = (
        df["name"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["test_type"].fillna("")
    )

    texts = df["combined_text"].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(df, f)

    print("Index built and saved.")

if __name__ == "__main__":
    build_index()