import chromadb
from chromadb.config import Settings
import uvicorn
from chromadb.server.fastapi import FastAPI

settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    allow_reset=True,
    is_persistent=True,
    anonymized_telemetry=False
)

app = FastAPI(settings).app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 