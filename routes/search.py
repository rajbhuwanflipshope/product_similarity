from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from qdrant_client.models import Filter, FieldCondition, Range

from model_loader import get_model
from qdrant_conn import get_client, COLLECTION_NAME

router = APIRouter()

# -------------------------
# Config
# -------------------------

TOP_K = 11
HNSW_EF = 16
SCORE_THRESHOLD = 0.60
DAYS_BACK = 45
TIMEOUT = 60


# -------------------------
# Init
# -------------------------

model = get_model()
client = get_client()

CUTOFF_TIME = int(
    (datetime.utcnow() - timedelta(days=DAYS_BACK)).timestamp()
)


# -------------------------
# Schema
# -------------------------

class SearchRequest(BaseModel):
    job_id: str
    title: str


# -------------------------
# Core Search
# -------------------------

async def find_similar_products(title: str):

    if not title:
        return []

    try:

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
            score_threshold=SCORE_THRESHOLD,
            with_payload=["_id"],
            search_params={"hnsw_ef": HNSW_EF},
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="lp_time",
                        range=Range(gte=CUTOFF_TIME)
                    )
                ]
            ),
            timeout=TIMEOUT,
        )

        results = results_obj.points

        return [
            {"_id": r.payload["_id"]}
            for r in results
            if r.payload and "_id" in r.payload
        ]

    except Exception as e:
        print("SEARCH ERROR:", str(e))
        raise e


# -------------------------
# Route
# -------------------------

@router.post("/sim_project/search")
async def search_products(request: SearchRequest):

    try:
        print(f"job_id {request.job_id} title {request.title}")
        similar_products = await find_similar_products(request.title)
       # print(f"job_id {request.job_id} title {request.title}")
        return {
            "job_id": request.job_id,
            "similar_products": similar_products
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
