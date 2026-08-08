# orchestrate existing modules

from pathlib import Path

from src.ingest import ingest_documents
from src.embeddings import generate_embeddings
from src.pinecone_client import upsert_embedded_chunks

SOURCE_ROOT = Path("sources")

chunks = ingest_documents(SOURCE_ROOT)
print(f"Loaded {len(chunks)} chunks")

embedded_chunks = generate_embeddings(chunks)
print(f"Generated {len(embedded_chunks)} embeddings")

response = upsert_embedded_chunks(embedded_chunks)
print("Pinecone upsert complete")
print(response)