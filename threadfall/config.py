from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "threadfall_memories")
    embed_dim: int = int(os.getenv("THREADFALL_EMBED_DIM", "128"))
    requested_device: str = os.getenv("THREADFALL_EMBED_DEVICE", "auto")
    save_path: Path = Path(os.getenv("THREADFALL_SAVE_PATH", "threadfall_save.json"))
    ollama_url: str = os.getenv("THREADFALL_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("THREADFALL_OLLAMA_MODEL", "llama3.2:latest")
    ollama_embed_model: str = os.getenv("THREADFALL_OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    ollama_timeout_seconds: int = int(os.getenv("THREADFALL_OLLAMA_TIMEOUT", "20"))
    ollama_enabled: bool = os.getenv("THREADFALL_OLLAMA_ENABLED", "true").lower() == "true"


SETTINGS = Settings()
