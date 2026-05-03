from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class DecisionProposal:
    action: str
    target: str
    reasoning: str
    confidence: float
    source: str


class OllamaDecisionEngine:
    def __init__(self, base_url: str, model: str, timeout_seconds: int, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.backend_name = "ollama"
        self.status = self._probe_status()

    def _probe_status(self) -> dict[str, str]:
        if not self.enabled:
            return {"available": "false", "detail": "disabled by configuration"}
        try:
            response = self._get_json(
                path="/api/tags",
                timeout=min(self.timeout_seconds, 10),
            )
        except Exception as exc:
            return {"available": "false", "detail": f"probe failed: {exc}"}

        names = {item.get("name", "") for item in response.get("models", [])}
        model_present = self.model in names
        detail = "ready" if model_present else f"model {self.model} not installed"
        return {"available": "true" if model_present else "false", "detail": detail}

    def is_available(self) -> bool:
        return self.status["available"] == "true"

    def propose_decision(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> tuple[DecisionProposal | None, str]:
        response, status = self.generate_json(prompt, schema=schema)
        if response is None:
            return None, status

        data = response
        try:
            proposal = DecisionProposal(
                action=str(data["action"]).strip(),
                target=str(data["target"]).strip(),
                reasoning=str(data["reasoning"]).strip(),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                source=self.model,
            )
        except Exception as exc:
            return None, f"response shape error: {exc}"
        return proposal, "ok"

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        if not self.is_available():
            return None, self.status["detail"]

        try:
            response = self._post_json(
                path="/api/generate",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema or "json",
                },
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            detail = f"ollama request failed: {exc}"
            return None, detail

        data = self._parse_json(str(response.get("response", "")).strip())
        if data is None:
            return None, f"invalid JSON response: {str(response.get('response', ''))[:240]}"
        return data, "ok"

    @staticmethod
    def _parse_json(output: str) -> dict[str, Any] | None:
        cleaned = output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http {exc.code}: {detail}") from exc

    def _get_json(self, path: str, timeout: int) -> dict[str, Any]:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http {exc.code}: {detail}") from exc


def propose_expedition_commit_supplies(
    engine: OllamaDecisionEngine,
    mission: dict[str, Any],
    world: dict[str, Any],
    *,
    max_commit: int,
) -> tuple[int | None, str]:
    """
    Ask the local model how many extra supplies to commit (0..max_commit) for this expedition.
    Returns (commit, status). On failure returns (None, reason) — caller should fall back to heuristics.
    """
    if max_commit <= 0:
        return 0, "no_commit_cap"
    if not engine.is_available():
        return None, str(engine.status.get("detail", "ollama unavailable"))

    schema = {
        "type": "object",
        "properties": {
            "commit": {"type": "integer"},
            "reasoning": {"type": "string"},
        },
        "required": ["commit", "reasoning"],
    }
    prompt = f"""
You are the expedition logistics AI for Threadfall. Reply with JSON only.

Choose integer commit in [0, {max_commit}] = extra supplies to risk this cycle (improves odds but is consumed).

Mission:
- type: {mission.get("mission_type", "standard")}
- difficulty: {mission.get("difficulty", "Unknown")}
- sector: {mission.get("sector", "")}
- objective: {mission.get("objective", "")}

World:
- supplies: {world.get("supplies", 0)}
- threat_clock: {world.get("threat_clock", 0)}
- sector_stability: {world.get("sector_stability", 0)}
- intel: {world.get("intel", 0)}

Return JSON: {{"commit": 0, "reasoning": "under 25 words"}}
""".strip()
    data, status = engine.generate_json(prompt, schema=schema)
    if data is None:
        return None, status
    try:
        c = int(data.get("commit", 0))
    except (TypeError, ValueError):
        return None, "commit not int"
    c = max(0, min(int(max_commit), c))
    return c, "ok"
