from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import torch


TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Embedder:
    dim: int
    device: str
    backend_name: str
    model_name: str
    status: dict[str, str]

    @classmethod
    def create(
        cls,
        dim: int,
        requested_device: str,
        ollama_enabled: bool,
        ollama_url: str,
        ollama_model: str,
        ollama_timeout_seconds: int,
    ) -> "Embedder":
        if ollama_enabled:
            remote = OllamaEmbedder.try_create(
                base_url=ollama_url,
                model=ollama_model,
                timeout_seconds=ollama_timeout_seconds,
            )
            if remote is not None:
                return remote

        return LocalHashEmbedder.create(dim=dim, requested_device=requested_device)

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass
class LocalHashEmbedder(Embedder):
    @classmethod
    def create(cls, dim: int, requested_device: str) -> "LocalHashEmbedder":
        if requested_device == "mps":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif requested_device == "cpu":
            device = "cpu"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        return cls(
            dim=dim,
            device=device,
            backend_name="local-hash",
            model_name="local-hash",
            status={"available": "true", "detail": "fallback embedder active"},
        )

    def encode(self, text: str) -> list[float]:
        tokens = TOKEN_RE.findall(text.lower())
        vector = torch.zeros(self.dim, dtype=torch.float32, device=self.device)

        if not tokens:
            return vector.cpu().tolist()

        for token in tokens:
            idx = hash(token) % self.dim
            vector[idx] += 1.0

        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) > 0.0:
            vector = vector / norm

        return vector.detach().cpu().tolist()


@dataclass
class OllamaEmbedder(Embedder):
    base_url: str
    timeout_seconds: int

    @classmethod
    def try_create(
        cls,
        base_url: str,
        model: str,
        timeout_seconds: int,
    ) -> "OllamaEmbedder" | None:
        status = cls._probe(base_url=base_url, model=model, timeout_seconds=timeout_seconds)
        if status["available"] != "true":
            return None

        sample = cls._embed_request(
            base_url=base_url,
            model=model,
            text="threadfall embedding probe",
            timeout_seconds=timeout_seconds,
        )
        vector = sample.get("embedding", [])
        if not vector:
            return None

        return cls(
            dim=len(vector),
            device="remote",
            backend_name="ollama",
            model_name=model,
            status=status,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
        )

    def encode(self, text: str) -> list[float]:
        response = self._embed_request(
            base_url=self.base_url,
            model=self.model_name,
            text=text,
            timeout_seconds=self.timeout_seconds,
        )
        embedding = response.get("embedding", [])
        if not embedding:
            raise RuntimeError("ollama embedding response missing vector")
        return [float(value) for value in embedding]

    @staticmethod
    def _probe(base_url: str, model: str, timeout_seconds: int) -> dict[str, str]:
        req = request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        try:
            with request.urlopen(req, timeout=min(timeout_seconds, 10)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return {"available": "false", "detail": f"probe failed: {exc}"}

        names = {item.get("name", "") for item in payload.get("models", [])}
        if model not in names:
            return {"available": "false", "detail": f"model {model} not installed"}
        return {"available": "true", "detail": "ready"}

    @staticmethod
    def _embed_request(
        base_url: str,
        model: str,
        text: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = request.Request(
            f"{base_url.rstrip('/')}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http {exc.code}: {detail}") from exc
