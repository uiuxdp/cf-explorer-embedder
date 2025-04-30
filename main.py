import chromadb
import pandas as pd
import requests
import os
import json
from tqdm import tqdm  # For progress bars
import time
import multiprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import socket  # Import the socket module
import subprocess  # Import the subprocess module

# Configuration
CHROMA_DB_PATH = "./chroma_db"
LM_STUDIO_URL = "http://10.90.115.176:1234/v1/embeddings"  # Your LM Studio embeddings endpoint
# LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"  # Your LM Studio embeddings endpoint
CSV_FILE_PATH = "Website_Sentiment_Looker - Website_Sentiment_Looker.csv"  # Replace with your CSV file path
TEXT_COLUMN = "Comments"  # Corrected to use the Comments column from your sample data
ID_COLUMN = None  # Replace with your ID column if you have one, otherwise None
COLLECTION_NAME = "documents_collection"
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8000
MAX_CONNECTION_ATTEMPTS = 10
CONNECTION_RETRY_DELAY = 1  # seconds

# Create the directory if it doesn't exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Define FastAPI app and models
app = FastAPI(title="Document Search API")

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class SearchResult(BaseModel):
    content: str
    similarity: float
    metadata: dict

class SearchResponse(BaseModel):
    results: List[SearchResult]

# Global variables for client and collection
client = None
collection = None

# Function to start ChromaDB server using subprocess
def start_chroma_server_process(path, host="0.0.0.0", port=8000):
    command = [
        "chroma",
        "run",
        "--path",
        path,
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"Starting ChromaDB server with command: {' '.join(command)}")
    process = subprocess.Popen(command)
    return process

def load_csv_data(file_path, text_column=None, id_column=None):
    """Load data from a CSV file and combine all columns for embedding."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Generate combined text from all columns for each row
    documents = []
    for _, row in df.iterrows():
        # Combine all column values into a single text, with column names as prefixes
        row_text = " ".join([f"{col}: {str(val).strip()}" for col, val in row.items()])
        documents.append(row_text)

    # Generate metadata from all columns
    metadatas = []
    for _, row in df.iterrows():
        metadata = {col: str(row[col]) for col in df.columns}
        metadatas.append(metadata)

    # Generate IDs if id_column is not provided
    if id_column and id_column in df.columns:
        ids = df[id_column].astype(str).tolist()
    else:
        ids = [f"doc{i}" for i in range(len(documents))]

    print(f"Loaded {len(documents)} documents")
    return documents, metadatas, ids


def get_embeddings_from_lm_studio(texts, api_url=LM_STUDIO_URL, batch_size=5):
    """Get embeddings from LM Studio API in batches."""
    headers = {
        "Content-Type": "application/json"
    }

    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]

        # Clean and validate the texts
        batch = [str(text).strip() for text in batch if text is not None]
        batch = [text for text in batch if len(text) > 0]

        if not batch:  # Skip empty batches
            continue
        data = {
            "input": batch,
            # "model": "text-embedding-nomic-embed-text-v1.5"
            "model": "text-embedding-mxbai-embed-large-v1"
        }

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(data))
                response.raise_for_status()
                result = response.json()

                # Extract embeddings from the response
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Error in batch {i//batch_size} after {max_retries} retries: {e}")
                    # Add None for each failed embedding in this batch
                    all_embeddings.extend([None] * len(batch))
                else:
                    print(f"Retry {retry_count}/{max_retries} for batch {i//batch_size}")
                    time.sleep(1)  # Wait a second before retrying

    # Remove any None embeddings and corresponding texts
    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None]
    valid_embeddings = [all_embeddings[i] for i in valid_indices]

    if len(valid_embeddings) < len(texts):
        print(f"Warning: Only generated {len(valid_embeddings)} embeddings for {len(texts)} texts")

    return valid_embeddings, valid_indices


def setup_chroma_db():
    """Set up ChromaDB client and attempt to connect with retries."""
    global client
    attempts = 0
    while attempts < MAX_CONNECTION_ATTEMPTS:
        try:
            # Updated for ChromaDB 1.0.5 - use HttpClient to connect to the server
            client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
            # Try a simple operation to check if the server is reachable
            client.heartbeat()
            print("Connected to ChromaDB server")

            # Always delete and recreate for consistent embedding size
            try:
                client.delete_collection(COLLECTION_NAME)
                print(f"Deleted old collection: {COLLECTION_NAME}")
            except Exception:
                pass

            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"Created new collection: {COLLECTION_NAME}")

            return client, collection
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts}/{MAX_CONNECTION_ATTEMPTS}: Error connecting to ChromaDB: {str(e)}")
            if attempts < MAX_CONNECTION_ATTEMPTS:
                time.sleep(CONNECTION_RETRY_DELAY)

    raise RuntimeError("Failed to connect to ChromaDB server after multiple retries.")


def add_documents_to_chroma(collection, documents, embeddings, metadatas, ids):
    """Add documents with embeddings to Chroma collection."""
    # Add in batches of 100
    batch_size = 100

    for i in tqdm(range(0, len(documents), batch_size), desc="Adding to Chroma"):
        end_idx = min(i + batch_size, len(documents))

        collection.add(
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

    print(f"Added {len(documents)} documents to Chroma")


def process_csv_to_chroma(csv_path, text_column, id_column=None):
    """Process CSV file and store in Chroma database."""
    # Load data from CSV
    documents, metadatas, ids = load_csv_data(csv_path, text_column, id_column)

    # Get embeddings from LM Studio
    print("Generating embeddings using LM Studio...")
    embeddings, valid_indices = get_embeddings_from_lm_studio(documents)

    # Filter documents, metadatas, and ids based on valid indices
    valid_documents = [documents[i] for i in valid_indices]
    valid_metadatas = [metadatas[i] for i in valid_indices]
    valid_ids = [ids[i] for i in valid_indices]

    # Set up Chroma and connect with retries
    client, collection = setup_chroma_db()

    # Add documents to Chroma
    add_documents_to_chroma(collection, valid_documents, embeddings, valid_metadatas, valid_ids)

    return client, collection

# Initialize database at startup
@app.on_event("startup")
async def startup_event():
    global client, collection, server_process
    # Start ChromaDB server in a separate process using the CLI
    print("Starting ChromaDB server...")
    server_process = start_chroma_server_process(CHROMA_DB_PATH, CHROMA_SERVER_HOST, CHROMA_SERVER_PORT)

    # Wait for a bit to allow the server to start (you might need to adjust this)
    time.sleep(5)

    # Process the CSV file and store in Chroma, with connection retries
    try:
        client, collection = process_csv_to_chroma(CSV_FILE_PATH, None, ID_COLUMN)
        print(f"Collection '{COLLECTION_NAME}' initialized with {collection.count()} documents")
    except RuntimeError as e:
        print(f"Error during ChromaDB initialization: {e}")
        # Consider how you want to handle this failure during startup
        # You might want to terminate the server process and exit the application
        if server_process and server_process.poll() is None:
            print("Terminating ChromaDB server process due to initialization failure.")
            server_process.terminate()
            server_process.wait()
        raise

# Define search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    # Get embeddings for the query
    query_embeddings, valid_indices = get_embeddings_from_lm_studio([request.query])

    if not valid_indices:
        raise HTTPException(status_code=400, detail="Failed to generate query embeddings")

    # Search the collection
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=request.n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Process results
    search_results = []
    for i in range(len(results["documents"][0])):
        # Convert distance to similarity (1 - distance for cosine)
        similarity = 1 - float(results["distances"][0][i])

        search_results.append(SearchResult(
            content=results["documents"][0][i],
            similarity=similarity,
            metadata=results["metadatas"][0][i]
        ))

    return SearchResponse(results=search_results)

if __name__ == "__main__":
    # Run the FastAPI server
    print("\nStarting search API server...")
    uvicorn.run(app, host="0.0.0.0", port=1111)