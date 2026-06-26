from datetime import datetime
from qdrant_client import QdrantClient

COLLECTION = "clothing_products"

client = QdrantClient(
    url="http://127.0.0.1:6335",
    api_key="H7kP2sM8Lx9QeA3N5R0CwZ4VYB6D1JtFUpoXiKrmvS"
)

offset = None

def to_ts(t):
    if isinstance(t, int):
        return t

    if t.endswith("Z"):
        t = t.replace("Z", "+00:00")

    return int(datetime.fromisoformat(t).timestamp())


while True:

    points, offset = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        offset=offset,
        with_payload=True
    )

    if not points:
        break

    for p in points:

        lp = p.payload.get("lp_time")

        if isinstance(lp, str):

            ts = to_ts(lp)

            client.set_payload(
                collection_name=COLLECTION,
                payload={"lp_time": ts},
                points=[p.id]
            )

            print("updated:", p.id)

    if offset is None:
        break

print("migration complete")
