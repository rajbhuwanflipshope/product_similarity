from qdrant_client import AsyncQdrantClient
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://127.0.0.1:6335"   
QDRANT_API_KEY = os.getenv("QDRANT_API")
INGEST_API_KEY = os.getenv("INGEST_API_KEY")

COLLECTION_NAME = "clothing_products"

client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

def get_client():
    return client

