# =====================================================
# CPU-ONLY & SILENT (MUST BE FIRST)
# =====================================================
import os
import sys
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from model_loader import get_model
from qdrant_conn import get_client, COLLECTION_NAME

router = APIRouter()

# =====================================================
# CONFIG (SEARCH TUNING)
# =====================================================

TOP_K = 11
HNSW_EF = 16
SCORE_THRESHOLD = 0.60
DAYS_BACK = 45
TIMEOUT = 60

# =====================================================
# INIT (loaded once per worker)
# =====================================================

model = get_model()
client = get_client()

CUTOFF_TIME = (
    datetime.utcnow() - timedelta(days=DAYS_BACK)
).isoformat()


# =====================================================
# SCHEMA
# =====================================================

class SearchRequest(BaseModel):
    job_id: str
    title: str


# =====================================================
# CORE SEARCH
# =====================================================

async def find_similar_products(title: str):
    if not title:
        return []

    # Offload CPU-bound encoding to a thread pool
    query_vector = await run_in_threadpool(
        model.encode,
        title,
        normalize_embeddings=True
    )
    query_vector = query_vector.tolist()

    results_obj = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        search_params={"hnsw_ef": HNSW_EF},
        score_threshold=SCORE_THRESHOLD,
        with_payload=["_id"],
        query_filter={
            "must": [
                {
                    "key": "lp_time",
                    "range": {"gte": CUTOFF_TIME}
                }
            ]
        },
        timeout=TIMEOUT,
    )
    results = results_obj.points

    return [
        {"_id": r.payload["_id"]}
        for r in results
    ]


# =====================================================
# ROUTE
# =====================================================

@router.post("/sim_project/search")
async def search_products(request: SearchRequest):
    try:
        similar_products = await find_similar_products(request.title)

        return {
            "job_id": request.job_id,
            "similar_products": similar_products
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
