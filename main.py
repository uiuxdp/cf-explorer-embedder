import os
import json
import time
import requests
import pandas as pd
import subprocess
from tqdm import tqdm
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import chromadb

# --- Config ---
CHROMA_DB_PATH = "./chroma_db"
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8004
LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"
STRAPI_API_URL = "http://localhost:1337/api/feedback-items?populate=source"
STRAPI_BASE_URL = "http://localhost:1337"
ID_COLUMN = None
MAX_CONNECTION_ATTEMPTS = 10
CONNECTION_RETRY_DELAY = 1
DOWNLOADS_DIR = "./downloads"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# --- FastAPI setup ---
app = FastAPI(title="Multi-Collection Search API")
client = None
collections_by_slug = {}
server_process = None

# --- Models ---
class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class SearchResult(BaseModel):
    content: str
    similarity: float
    metadata: dict

class SearchResponse(BaseModel):
    results: List[SearchResult]

# --- Start ChromaDB subprocess ---
def start_chroma_server(path, host, port):
    command = ["chroma", "run", "--path", path, "--host", host, "--port", str(port)]
    print(f"Starting ChromaDB: {' '.join(command)}")
    return subprocess.Popen(command)

# --- CSV Loader ---
def load_csv(file_path):
    df = pd.read_csv(file_path)
    documents = [" ".join([f"{col}: {str(val).strip()}" for col, val in row.items()]) for _, row in df.iterrows()]
    metadatas = [{col: str(row[col]) for col in df.columns} for _, row in df.iterrows()]
    ids = [f"doc{i}" for i in range(len(documents))]
    return documents, metadatas, ids

# --- LM Studio embedding ---
def get_embeddings(texts, api_url=LM_STUDIO_URL, batch_size=5):
    headers = {"Content-Type": "application/json"}
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = [str(t).strip() for t in texts[i:i+batch_size] if t and len(t.strip()) > 0]
        if not batch: continue
        # text-embedding-mxbai-embed-large-v1
        data = {"input": batch, "model": "text-embedding-nomic-embed-text-v1.5"}
        retries = 3
        while retries > 0:
            try:
                res = requests.post(api_url, headers=headers, data=json.dumps(data))
                res.raise_for_status()
                embeddings.extend([r["embedding"] for r in res.json()["data"]])
                break
            except Exception as e:
                retries -= 1
                time.sleep(1)
                if retries == 0:
                    print(f"Failed embedding batch: {e}")
                    embeddings.extend([None] * len(batch))
    valid = [i for i, e in enumerate(embeddings) if e is not None]
    return [embeddings[i] for i in valid], valid

# --- ChromaDB Setup ---
def setup_chroma():
    for attempt in range(MAX_CONNECTION_ATTEMPTS):
        try:
            client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
            client.heartbeat()
            print("Connected to ChromaDB.")
            return client
        except Exception as e:
            print(f"Attempt {attempt+1}: {e}")
            time.sleep(CONNECTION_RETRY_DELAY)
    raise RuntimeError("Could not connect to ChromaDB.")

# --- Index single CSV into Chroma collection ---
def index_csv_to_collection(client, file_path, slug):
    documents, metadatas, ids = load_csv(file_path)
    embeddings, valid_idx = get_embeddings(documents)
    valid_docs = [documents[i] for i in valid_idx]
    valid_metas = [metadatas[i] for i in valid_idx]
    valid_ids = [ids[i] for i in valid_idx]

    try:
        client.delete_collection(slug)
    except:
        pass

    collection = client.create_collection(name=slug, metadata={"hnsw:space": "cosine"})
    for i in tqdm(range(0, len(valid_docs), 100), desc=f"Adding to {slug}"):
        collection.add(
            documents=valid_docs[i:i+100],
            embeddings=embeddings[i:i+100],
            metadatas=valid_metas[i:i+100],
            ids=valid_ids[i:i+100]
        )
    return collection

# --- Load all CSVs from Strapi API ---
def index_all_sources():
    global collections_by_slug
    res = requests.get(STRAPI_API_URL)
    items = res.json().get("data", [])
    for item in items:
        slug = item.get("slug")
        src = item.get("source")

        if not slug or not src or not src.get("url") or not src.get("name"):
            print(f"Skipping invalid entry: {item}")
            continue

        url = f"{STRAPI_BASE_URL}{src['url']}"
        filename = os.path.join(DOWNLOADS_DIR, src["name"])
        print(f"Downloading {url}...")

        try:
            with open(filename, "wb") as f:
                f.write(requests.get(url).content)
            collections_by_slug[slug] = index_csv_to_collection(client, filename, slug)
            print(f"Indexed {slug}")
        except Exception as e:
            print(f"Error indexing {slug}: {e}")


# --- Startup: launch Chroma, connect, and index all ---
@app.on_event("startup")
async def on_startup():
    global client, server_process
    print("Launching ChromaDB server...")
    server_process = start_chroma_server(CHROMA_DB_PATH, CHROMA_SERVER_HOST, CHROMA_SERVER_PORT)
    time.sleep(5)
    client = setup_chroma()
    index_all_sources()

@app.on_event("shutdown")
async def on_shutdown():
    if server_process and server_process.poll() is None:
        print("Shutting down ChromaDB...")
        server_process.terminate()
        server_process.wait()

# --- Search Endpoint ---
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, slug: str = Query(..., description="Collection slug")):
    collection = collections_by_slug.get(slug)
    if not collection:
        raise HTTPException(status_code=404, detail=f"Collection '{slug}' not found")

    query_embeddings, valid_idx = get_embeddings([request.query])
    if not valid_idx:
        raise HTTPException(status_code=400, detail="Embedding failed")

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=request.n_results,
        include=["documents", "metadatas", "distances"]
    )

    return SearchResponse(results=[
        SearchResult(
            content=results["documents"][0][i],
            similarity=1 - results["distances"][0][i],
            metadata=results["metadatas"][0][i]
        )
        for i in range(len(results["documents"][0]))
    ])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)
