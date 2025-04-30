import os
import pandas as pd
import requests
import json
from tqdm import tqdm
import time
import chromadb
from fastapi import UploadFile

CHROMA_DB_BASE_PATH = "./chroma_db"
LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"


def load_csv_data(file_path):
    df = pd.read_csv(file_path)

    documents = []
    for _, row in df.iterrows():
        row_text = " ".join([f"{col}: {str(val).strip()}" for col, val in row.items()])
        documents.append(row_text)

    metadatas = []
    for _, row in df.iterrows():
        metadata = {col: str(row[col]) for col in df.columns}
        metadatas.append(metadata)

    ids = [f"doc{i}" for i in range(len(documents))]
    return documents, metadatas, ids


def get_embeddings_from_lm_studio(texts, batch_size=5):
    headers = {"Content-Type": "application/json"}
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        batch = [str(text).strip() for text in batch if text and len(text) > 0]

        if not batch:
            continue

        data = {
            "input": batch,
            "model": "text-embedding-nomic-embed-text-v1.5"
        }

        for attempt in range(3):
            try:
                response = requests.post(LM_STUDIO_URL, headers=headers, data=json.dumps(data))
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend([item["embedding"] for item in result["data"]])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed batch {i//batch_size}: {e}")
                    all_embeddings.extend([None] * len(batch))
                else:
                    time.sleep(1)

    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None]
    valid_embeddings = [all_embeddings[i] for i in valid_indices]
    return valid_embeddings, valid_indices


def add_documents_to_chroma(collection, documents, embeddings, metadatas, ids):
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding to Chroma"):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )


def index_agent_csv(slug: str, files: list[UploadFile]):
    csv_folder = f"./csv_data/{slug}"
    os.makedirs(csv_folder, exist_ok=True)

    all_docs, all_meta, all_ids = [], [], []
    for file in files:
        path = os.path.join(csv_folder, file.filename)
        with open(path, "wb") as f:
            content = file.file.read()
            f.write(content)

        docs, meta, ids = load_csv_data(path)
        all_docs.extend(docs)
        all_meta.extend(meta)
        all_ids.extend(ids)

    embeddings, valid_indices = get_embeddings_from_lm_studio(all_docs)
    valid_docs = [all_docs[i] for i in valid_indices]
    valid_meta = [all_meta[i] for i in valid_indices]
    valid_ids = [all_ids[i] for i in valid_indices]

    chroma_path = os.path.join(CHROMA_DB_BASE_PATH, slug)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(f"{slug}_collection", metadata={"hnsw:space": "cosine"})

    add_documents_to_chroma(collection, valid_docs, embeddings, valid_meta, valid_ids)
    return len(valid_docs)


def search_agent_documents(slug: str, query: str, n_results: int):
    chroma_path = os.path.join(CHROMA_DB_BASE_PATH, slug)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(f"{slug}_collection")

    query_embeddings, valid_indices = get_embeddings_from_lm_studio([query])
    if not valid_indices:
        raise ValueError("Embedding generation failed.")

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "content": results["documents"][0][i],
            "similarity": 1 - results["distances"][0][i],
            "metadata": results["metadatas"][0][i]
        }
        for i in range(len(results["documents"][0]))
    ]


def list_agents():
    return [name for name in os.listdir(CHROMA_DB_BASE_PATH) if os.path.isdir(os.path.join(CHROMA_DB_BASE_PATH, name))]