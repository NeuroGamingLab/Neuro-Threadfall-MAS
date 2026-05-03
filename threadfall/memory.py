from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from threadfall.config import SETTINGS


@dataclass
class MemoryRecord:
    id: str
    text: str
    score: float
    payload: dict[str, Any]


class LocalMemoryStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.backend_name = "local"
        self._items: list[dict[str, Any]] = []

    def save(self, text: str, payload: dict[str, Any]) -> str:
        item_id = str(uuid.uuid4())
        self._items.append(
            {
                "id": item_id,
                "text": text,
                "vector": self.embedder.encode(text),
                "payload": payload,
            }
        )
        return item_id

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        qvec = self.embedder.encode(query)

        def score(item: dict[str, Any]) -> float:
            return sum(a * b for a, b in zip(qvec, item["vector"]))

        ranked = sorted(self._items, key=score, reverse=True)[:limit]
        return [
            MemoryRecord(
                id=item["id"],
                text=item["text"],
                score=score(item),
                payload=item["payload"],
            )
            for item in ranked
        ]


class QdrantMemoryStore:
    def __init__(self, client: QdrantClient, collection: str, embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.backend_name = "qdrant"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = self.client.collection_exists(self.collection)
        if not existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=SETTINGS.embed_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(self.collection)
        current_size = info.config.params.vectors.size
        if current_size != self.embedder.dim:
            self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.embedder.dim,
                    distance=models.Distance.COSINE,
                ),
            )

    def save(self, text: str, payload: dict[str, Any]) -> str:
        item_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=item_id,
            vector=self.embedder.encode(text),
            payload={"text": text, **payload, "created_at": int(time.time())},
        )
        self.client.upsert(collection_name=self.collection, points=[point])
        return item_id

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.encode(query),
            limit=limit,
        ).points

        return [
            MemoryRecord(
                id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=float(hit.score or 0.0),
                payload=dict(hit.payload),
            )
            for hit in hits
        ]


def create_memory_store(embedder):
    try:
        client = QdrantClient(url=SETTINGS.qdrant_url)
        client.get_collections()
        return QdrantMemoryStore(
            client=client,
            collection=SETTINGS.qdrant_collection,
            embedder=embedder,
        )
    except Exception:
        return LocalMemoryStore(embedder)
