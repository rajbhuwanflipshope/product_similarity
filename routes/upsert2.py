import os
import uuid
import traceback
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


# -------------------------
# Core Upsert Logic
# -------------------------
async def upsert_logic(doc: Product):
    point_id = pid(doc.id)

    try:
        # Check if point exists
        existing = await client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )

        if existing:
            await client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"lp_time": doc.lp_time},
                points=[point_id],
            )
            return {"status": "updated", "_id": doc.id}

        # Encode title → vector
        vector = await run_in_threadpool(
            model.encode,
            doc.title,
            normalize_embeddings=True
        )
        vector = vector.tolist()

        # Insert new point
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=doc.model_dump(by_alias=True),
                )
            ],
        )

        return {"status": "inserted", "_id": doc.id}

    except Exception as e:
        print("\n----- UPSERT ERROR -----")
        traceback.print_exc()
        print("------------------------\n")
        raise e


# -------------------------
# API Route
# -------------------------
@router.post("/sim_project/insert")
async def insert_product(p: Product):
    try:
        return await upsert_logic(p)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Health Endpoint
# -------------------------
@router.get("/sim_project/health")
def health():
    return {"status": "ok"}
