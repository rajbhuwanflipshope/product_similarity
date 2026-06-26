import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from qdrant_client.models import PointStruct

from model_loader import get_model
from qdrant_conn import get_client, COLLECTION_NAME

router = APIRouter()

model = get_model()
client = get_client()


# -------------------------
# Schema
# -------------------------

class Product(BaseModel):
    id: str = Field(alias="_id")
    title: str
    lp_time: str

    class Config:
        populate_by_name = True


# -------------------------
# Helpers
# -------------------------

def pid(v: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, v))


def to_timestamp(t: str) -> int:
    """
    Convert ISO8601 time to unix timestamp.
    Handles Zulu time (Z).
    """

    if t.endswith("Z"):
        t = t.replace("Z", "+00:00")

    return int(datetime.fromisoformat(t).timestamp())


# -------------------------
# Core Upsert Logic
# -------------------------

async def upsert_logic(doc: Product):

    point_id = pid(doc.id)

    try:

        existing = await client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )

        ts = to_timestamp(doc.lp_time)

        if existing:

            await client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"lp_time": ts},
                points=[point_id],
            )

            return {
                "status": "updated",
                "_id": doc.id
            }

        # encode embedding
        vector = await run_in_threadpool(
            model.encode,
            doc.title,
            normalize_embeddings=True
        )

        vector = vector.tolist()

        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "_id": doc.id,
                        "title": doc.title,
                        "lp_time": ts
                    },
                )
            ],
        )

        return {
            "status": "inserted",
            "_id": doc.id
        }

    except Exception as e:
        print("UPSERT ERROR:", str(e))
        raise e


# -------------------------
# Routes
# -------------------------

@router.post("/sim_project/insert")
async def insert_product(p: Product):

    try:
        return await upsert_logic(p)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/sim_project/health")
def health():
    return {"status": "ok"}
