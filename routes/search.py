import time
import uuid
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_conn import get_client, COLLECTION_NAME

router = APIRouter()

# -------------------------
# Config
# -------------------------

TOP_K = 11
SCORE_THRESHOLD = 0.60
DAYS_BACK = 45
TIMEOUT = 60

client = get_client()

# -------------------------
# Helpers
# -------------------------

@lru_cache(maxsize=500_000)
def get_qdrant_point_id(mongo_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, mongo_id))

# -------------------------
# Schema
# -------------------------

class SearchRequest(BaseModel):
    job_id: str
    id: str = Field(alias="_id")

    class Config:
        populate_by_name = True

# -------------------------
# Core Search
# -------------------------

async def find_similar_products(product_id: str):
    if not product_id:
        return []

    point_id = get_qdrant_point_id(product_id)
    cutoff = ((int(time.time()) // 3600) * 3600) - DAYS_BACK * 86400

    try:
        results_obj = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=point_id,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD,
            with_payload=["_id"],
            search_params={"hnsw_ef": 16},
            query_filter=Filter(
                must=[FieldCondition(key="lp_time", range=Range(gte=cutoff))]
            ),
            timeout=TIMEOUT,
        )
        results = results_obj.points
    except Exception as e:
        msg = str(e)
        if "No point with id" in msg or "NOT_FOUND" in msg:
            return []
        raise

    return [
        {"_id": payload["_id"]}
        for r in results
        if (payload := r.payload) and payload.get("_id") != product_id
    ]

# -------------------------
# Route
# -------------------------

@router.post("/sim_project/search")
async def search_products(request: SearchRequest):
    try:
        return {
            "job_id": request.job_id,
            "similar_products": await find_similar_products(request.id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))