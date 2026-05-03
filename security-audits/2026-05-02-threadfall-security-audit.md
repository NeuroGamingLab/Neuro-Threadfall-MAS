# Security Audit — Threadfall (Neuro-Threadfall-MAS)

| Field | Value |
|--------|--------|
| **Date** | 2026-05-02 |
| **Scope** | `app.py`, `threadfall/*.py`, `rl/`, `scripts/`, `Dockerfile`, `docker-compose.yml`, `run.sh`, `requirements*.txt` |
| **Method** | Static review (Security Audit Guard, full technical steps 1–12) |
| **Canada / sovereignty** | Not in scope |
| **Overall rating** | **Caution** |

**Summary:** No obvious remote code execution or classic SQL/shell injection in scanned application Python. Main concerns: **integrity of encounter payloads** (`?enc=` → session state), **Streamlit deployment posture** (no built-in auth), **HTML / `unsafe_allow_html` / `components.html`**, **Docker bind mounts**, and **unpinned base images**. Appropriate for **local / lab** use; not “safe by default” if the app is exposed on the open internet without controls.

---

## 1. Injection and deserialization

| Item | Rating | Notes |
|------|--------|--------|
| `eval` / `exec` / `pickle` / `subprocess` / `shell=True` | **Safe** | Not used in application code paths reviewed. |
| `json.loads` | **Caution** | Campaign state from `SETTINGS.save_path`; Ollama/Qdrant HTTP responses in `llm.py` / `embedder.py`. Trust boundaries are filesystem + configured services. |
| **`enc` query parameter** | **Caution** | `app.py` decodes `st.query_params["enc"]` via `decode_encounter_query` (base64 + JSON) into `pending_encounter`. Payload is **not signed**. Use only applies when `mission_id` matches current mission, but **guessable/leaked mission IDs** or **reflected links** can affect integrity when the user resolves. Mitigate with **HMAC**, server-side session storage, or strict schema + size limits. |

---

## 2. XSS and HTML surfaces

| Item | Rating | Notes |
|------|--------|--------|
| `unsafe_allow_html=True` (`threadfall/ui.py`) | **Caution** | Fine if all strings are server-controlled; re-audit if **user or LLM text** is interpolated into HTML without escaping. |
| `streamlit.components.v1.html` (encounters) | **Caution** | Generated HTML/JS in `encounters.py`; validate mission/context fields if any source becomes untrusted. |

---

## 3. Network and trust boundaries

| Item | Rating | Notes |
|------|--------|--------|
| `THREADFALL_OLLAMA_URL`, `QDRANT_URL` | **Caution** | Environment-controlled endpoints; misconfiguration can point HTTP clients at unintended hosts (SSRF class in hostile env). |
| TLS | **Caution** | Defaults are HTTP (normal on localhost); use TLS + private network for remote stacks. |

---

## 4. Secrets and sensitive data

| Item | Rating | Notes |
|------|--------|--------|
| Hardcoded cloud API keys | **Safe** | None observed in reviewed paths. |
| `.streamlit/secrets.toml` | **Caution** | Listed in `.gitignore` — keep secrets out of git. |
| Memory / save JSON | **Caution** | May contain **PII if users paste it** — organizational policy, not enforced in code. |

---

## 5. Filesystem and containers

| Item | Rating | Notes |
|------|--------|--------|
| `THREADFALL_SAVE_PATH` | **Caution** | Configurable read/write path; rely on OS permissions. |
| `docker-compose` `.:/app` | **Caution** | Host bind mount — convenient for dev; container compromise can alter host project tree. |
| `qdrant/qdrant:latest` | **Caution** | Floating tag — prefer **digest-pinned** images for reproducible, auditable deploys. |

---

## 6. Authentication and multi-tenant

| Item | Rating | Notes |
|------|--------|--------|
| Streamlit default | **Caution** | Anyone who can reach the port can use the app as that OS user. Use **VPN**, **reverse proxy + auth**, or hosted platform controls for internet exposure. |

---

## 7. Availability

| Item | Rating | Notes |
|------|--------|--------|
| Autoplay / ML swarm / Ollama | **Caution** | Long synchronous work can starve the Streamlit worker — acceptable for lab; cap or isolate for shared hosting. |

---

## 8. Dependencies

| Item | Rating | Notes |
|------|--------|--------|
| Supply chain | **Caution** | Use lockfiles / pinning and periodic review for production. |

---

## Rating scale

- **Safe:** No material issues for the stated use (e.g. trusted single-user local dev).
- **Caution:** Addressable with config, deployment, or small code changes; document assumptions.
- **Risky:** Would require major controls before internet-facing or regulated data — **not** assigned for local prototype scope; becomes **Risky** if exposed widely with no auth and unsigned `enc`.

---

## Recommended actions (priority)

1. **Sign or strictly validate** encounter payloads from `?enc=` (HMAC, or server-only session state).
2. **Document** threat model: single operator, trusted browser; do not expose Streamlit without auth.
3. **Pin** container images by digest; reduce bind-mount scope in production.
4. **Pass** each `unsafe_allow_html` and encounter HTML path whenever user/LLM strings reach markup.
5. For public demos: **TLS + auth** in front; keep Ollama/Qdrant on private networks.

---

## References

- Internal: `threadfall/game.py` (`resolve_current_mission`, `persist_state`), `app.py` (`enc` query handling), `threadfall/encounters.py` (`decode_encounter_query`).
- GitHub Mermaid / docs unrelated to this audit artifact.

_Filed under `security-audits/` for version control alongside the codebase._
