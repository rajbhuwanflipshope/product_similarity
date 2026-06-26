from fastapi import FastAPI
from routes.upsert import router as upsert_router
from routes.search import router as search_router

app = FastAPI(title="Vector API")

app.include_router(upsert_router)
app.include_router(search_router)
