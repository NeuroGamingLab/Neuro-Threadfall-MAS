from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


def _badge(label: str, tone: str = "neutral") -> str:
    styles = {
        "neutral": ("#E6EDF3", "#253041"),
        "info": ("#C6E6FF", "#16324F"),
        "success": ("#D9FBE8", "#173B2B"),
        "warning": ("#FFF0C2", "#4A3410"),
        "danger": ("#FFD7D7", "#4A1F24"),
        "accent": ("#DCCFFF", "#33214D"),
    }
    fg, bg = styles.get(tone, styles["neutral"])
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.78rem;font-weight:600;margin-right:0.35rem;'>"
        f"{label}</span>"
    )


def _severity_tone(value: str) -> str:
    mapping = {
        "Low": "info",
        "Elevated": "warning",
        "Severe": "danger",
        "Terminal": "danger",
        "Critical": "danger",
        "Active": "info",
        "Player Victory": "success",
        "Faction Victory": "accent",
        "Player Defeat": "danger",
        "Stalemate": "warning",
        "Success": "success",
        "Compromised": "danger",
        "Ready": "success",
        "Fallback": "warning",
    }
    return mapping.get(value, "neutral")


def _delta_tone(value: int) -> str:
    if value > 0:
        return "success"
    if value < 0:
        return "danger"
    return "neutral"


def _render_delta_badges(items: list[tuple[str, int]]) -> None:
    html = "".join(_badge(f"{label} {value:+d}", _delta_tone(value)) for label, value in items)
    st.markdown(html, unsafe_allow_html=True)


def render_status_cards(services) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Embed Device", services.embedder.device.upper())
    col2.metric("Memory Backend", services.memory.backend_name.upper())
    col3.metric("Vector Size", services.embedder.dim)

    st.info(
        "On macOS Apple Silicon, run the app natively to use the MPS GPU path. "
        "When the app runs inside Docker on Mac, the app should be treated as CPU-first."
    )
    status = services.decision_engine.status
    label = "Ready" if services.decision_engine.is_available() else "Fallback"
    st.markdown(
        _badge(f"Embedding {services.embedder.backend_name.upper()}", "info")
        + _badge(services.embedder.model_name, "accent"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Embedding model: {services.embedder.model_name} · {services.embedder.backend_name} · {services.embedder.status['detail']}"
    )
    st.markdown(
        _badge(f"Decision {label}", _severity_tone(label))
        + _badge(services.decision_engine.model, "accent"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Decision model: {services.decision_engine.model} · {label} · {status['detail']}"
    )


def _topology_fingerprint(*, sector_control, factions) -> str:
    """Stable hash of everything that affects the map visuals (avoids rebuilding the figure on unrelated reruns)."""
    from threadfall.game import current_sector_graph

    g = current_sector_graph()
    edges = sorted(tuple(sorted((a, b))) for a in g for b in g[a] if a < b)
    ctrl: dict[str, dict[str, int | str]] = {}
    for sector in sorted((sector_control or {}).keys()):
        info = (sector_control or {})[sector]
        ctrl[sector] = {
            "controller": str(info.get("controller", "Neutral")),
            "heat": int(info.get("heat", 0) or 0),
            "stability": int(info.get("stability", 0) or 0),
        }
    activity = st.session_state.get("sector_activity", {}) or {}
    act_slim = {k: activity[k] for k in sorted(activity.keys())}
    names = sorted({f.get("name", "") for f in (factions or [])})
    payload = {
        "cycle": int(st.session_state.get("world_state", {}).get("cycle", 0)),
        "edges": edges,
        "ctrl": ctrl,
        "activity": act_slim,
        "factions": names,
    }
    rival_bits: list[dict[str, Any]] = []
    for ag in sorted(st.session_state.get("ai_agents", []), key=lambda x: str(x.get("name", ""))):
        nm = str(ag.get("name", ""))
        if not nm:
            continue
        sec = ag.get("last_sector") or _infer_rival_last_sector_from_history(nm)
        rival_bits.append(
            {
                "name": nm,
                "sector": str(sec) if sec else "",
                "cycle": ag.get("last_cycle"),
                "intel": int(ag.get("intel", 0) or 0),
                "arts": int(ag.get("artifacts", 0) or 0),
            }
        )
    payload["rivals"] = rival_bits
    tail = st.session_state.get("activities", [])[-14:]
    payload["act_tail"] = [
        {"c": x.get("cycle"), "a": x.get("action"), "t": x.get("target"), "f": x.get("faction")} for x in tail
    ]

    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _infer_rival_last_sector_from_history(actor_name: str) -> str | None:
    """Best-effort last sector for a rival (older saves before last_sector was stored)."""
    from threadfall.game import SECTORS

    for item in reversed(st.session_state.get("history", [])):
        if item.get("actor") != actor_name or item.get("type") != "ai_expedition":
            continue
        sec = item.get("sector")
        if isinstance(sec, str) and sec in SECTORS:
            return sec
    return None


def _rival_unit_position(base_u: np.ndarray, *, slot: int, n_same: int) -> np.ndarray:
    """Place rival marker just outside sector node; spread siblings along a local tangent."""
    u = np.asarray(base_u, dtype=float)
    u = u / (float(np.linalg.norm(u)) or 1.0)
    aux = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(u, aux))) > 0.9:
        aux = np.array([1.0, 0.0, 0.0], dtype=float)
    t = np.cross(u, aux)
    t = t / (float(np.linalg.norm(t)) or 1.0)
    if n_same <= 1:
        off = 0.0
    else:
        mid = (n_same - 1) / 2.0
        off = 0.11 * (float(slot) - mid)
    raw = u + off * t
    raw = raw / (float(np.linalg.norm(raw)) or 1.0)
    return raw * 1.125


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        return f"rgba(170,188,235,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _dedupe_edge_candidates(cands: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    best: dict[tuple[int, str], str] = {}
    for cy, kind, blurb in cands:
        key = (int(cy), str(kind))
        prev = best.get(key, "")
        if len(blurb) >= len(prev):
            best[key] = blurb
    return [(cy, k, best[(cy, k)]) for cy, k in sorted(best.keys())]


def _globe_edge_kind_priority(kind: str) -> int:
    order = (
        "rewire_cut",
        "rewire_bridge",
        "sabotage",
        "sabotage_spill",
        "extract",
        "extract_spill",
        "research",
        "research_spill",
        "fortify",
        "fortify_spill",
        "player_fail",
        "ai_fail",
        "player_success",
        "ai_success",
        "event",
        "resupply",
    )
    if kind in order:
        return 100 - order.index(kind)
    return 10


def _globe_collect_edge_candidates(
    a: str,
    b: str,
    *,
    current_cycle: int,
    edge_window: int,
    sector_activity: dict[str, Any],
    activities_recent: list[dict[str, Any]],
) -> list[tuple[int, str, str]]:
    """(cycle, visual_kind, hover_blurb) for corridor endpoints + logged faction actions."""
    cand: list[tuple[int, str, str]] = []
    for sector in (a, b):
        act = sector_activity.get(sector) or {}
        kind = str(act.get("kind", ""))
        if not kind:
            continue
        cy = int(act.get("cycle", -999))
        if current_cycle - cy > edge_window:
            continue
        detail = str(act.get("detail", "")).strip()
        loc = f"[{sector}]"
        cand.append(
            (cy, kind, f"{loc} <b>{kind}</b>" + (f" · {detail}" if detail else "")),
        )

    for act in activities_recent:
        cy = int(act.get("cycle", -1))
        if cy < 0 or current_cycle - cy > edge_window:
            continue
        faction = str(act.get("faction", "?"))
        action = str(act.get("action", ""))
        target = str(act.get("target", ""))
        if action == "rewire":
            parts = target.split("|", 2)
            if len(parts) != 3 or parts[0] not in ("cut", "bridge"):
                continue
            p1, p2 = sorted((parts[1], parts[2]))
            if (p1, p2) != tuple(sorted((a, b))):
                continue
            rk = f"rewire_{parts[0]}"
            cand.append((cy, rk, f"<b>{faction}</b> · rewire {parts[0]} · {p1} ↔ {p2}"))
        elif action in ("extract", "research", "fortify"):
            if target not in (a, b):
                continue
            cand.append((cy, action, f"<b>{faction}</b> · {action} @ {target}"))

    return _dedupe_edge_candidates(cand)


def _globe_pick_edge_signal(candidates: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda t: (t[0], _globe_edge_kind_priority(t[1])))


def _globe_style_corridor_edge(
    a: str,
    b: str,
    *,
    current_cycle: int,
    edge_window: int,
    sector_activity: dict[str, Any],
    activities_recent: list[dict[str, Any]],
    sector_control: dict[str, Any],
    kind_hex: dict[str, str],
) -> tuple[str, int, str]:
    """Line rgba, width, hover html."""
    cands = _globe_collect_edge_candidates(
        a,
        b,
        current_cycle=current_cycle,
        edge_window=edge_window,
        sector_activity=sector_activity,
        activities_recent=activities_recent,
    )
    sig = _globe_pick_edge_signal(cands)
    ha = int((sector_control or {}).get(a, {}).get("heat", 0) or 0)
    hb = int((sector_control or {}).get(b, {}).get("heat", 0) or 0)
    mean_h = (ha + hb) / 2.0

    base_hover = f"<b>Corridor</b><br>{a}<br>↔<br>{b}<br>Heat {ha} / {hb}"

    if sig is None:
        if mean_h >= 62:
            return (
                _hex_to_rgba("#FF9966", 0.34 + min(0.22, (mean_h - 62) / 80.0)),
                4,
                base_hover + "<br><br><i>High sector heat — elevated transit risk.</i>",
            )
        return (
            "rgba(118,132,168,0.36)",
            3,
            base_hover + "<br><br><i>No recent corridor traffic in the live window.</i>",
        )

    cy, kind, blurb = sig
    hx = kind_hex.get(kind, "#AAB8DC")
    age = max(0, int(current_cycle) - int(cy))
    alpha = 0.9 if age <= 0 else 0.72 if age <= 1 else 0.52
    color = _hex_to_rgba(hx, alpha)
    if kind.startswith("rewire"):
        width = 7
    elif "spill" in kind:
        width = 5
    elif kind in ("player_fail", "ai_fail"):
        width = 6
    else:
        width = 6

    lines = [base_hover, "<br><b>Live corridor signal</b>", blurb, f"<br><small>Strongest cue · cycle {cy} · age {age}</small>"]
    if len(cands) > 1:
        extra = [f"• {c[2]}" for c in sorted(cands, key=lambda x: (-x[0], -_globe_edge_kind_priority(x[1])))[1:4]]
        lines.append("<br><i>Also seen:</i><br>" + "<br>".join(extra))
    return color, width, "".join(lines)


def _build_topology_figure(*, sector_control, factions) -> go.Figure:
    from threadfall.game import current_sector_graph

    # Stable layout (hand-tuned) so users build a mental map.
    pos = {
        "Archive Chamber Theta": (-1.8, 1.2),
        "Mnemonic Scriptorium": (-1.2, 0.3),
        "Signal Orchard": (-1.6, -0.2),
        "Mirror Fault Atrium": (-0.6, 0.9),
        "Inversion Causeway": (0.1, 0.5),
        "Vault Spine 3": (-0.2, -0.1),
        "Thermal Lens Cathedral": (0.6, 1.1),
        "Obsidian Switchyard": (0.6, 0.0),
        "Flooded Transit Gallery": (0.8, -0.8),
        "Tide-Locked Rotunda": (0.0, -1.2),
        "Resonance Stairwell": (-0.8, -0.9),
        "Null Pump Station": (-0.7, -0.35),
        "Gravimetric Choir": (1.2, 0.6),
    }

    controllers = {f["name"] for f in (factions or [])}
    palette = {
        "Neutral": "#8A94A6",
        "Helios Prospectors": "#FFCC6A",
        "Glass Archive": "#7EE7FF",
        "Morrow Division": "#B38CFF",
    }
    default_colors = ["#FFCC6A", "#7EE7FF", "#B38CFF", "#FF5B79", "#D4FF8A", "#DCCFFF"]
    for idx, name in enumerate(sorted(controllers)):
        palette.setdefault(name, default_colors[idx % len(default_colors)])

    graph = current_sector_graph()
    edge_x, edge_y = [], []
    mid_x, mid_y, mid_hover = [], [], []
    seen = set()
    for a, nbrs in graph.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            ia = (sector_control or {}).get(a, {})
            ib = (sector_control or {}).get(b, {})
            ha = int(ia.get("heat", 0) or 0)
            hb = int(ib.get("heat", 0) or 0)
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            mid_hover.append(
                f"<b>Corridor</b><br>{a}<br>↔<br>{b}<br>"
                f"<span style='opacity:0.85'>Combined heat {ha + hb} · "
                f"Agents can rewire this link (cut/bridge) to change spillover paths.</span>"
            )

    line_spline = dict(shape="spline", smoothing=0.65)
    edge_glow = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=9, color="rgba(98, 118, 200, 0.14)", **line_spline),
        hoverinfo="skip",
        showlegend=False,
    )
    edge_core = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2.2, color="rgba(170, 188, 235, 0.55)", **line_spline),
        hoverinfo="skip",
        showlegend=False,
    )
    edge_hit = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=18, color="rgba(255,255,255,0.06)", line=dict(width=0)),
        hoverinfo="text",
        hovertext=mid_hover,
        showlegend=False,
    )

    activity = st.session_state.get("sector_activity", {}) or {}
    current_cycle = int(st.session_state.get("world_state", {}).get("cycle", 0))
    active_window = 1

    kind_outline = {
        "extract": "#FFCC6A",
        "extract_spill": "#FFCC6A",
        "research": "#7EE7FF",
        "research_spill": "#7EE7FF",
        "fortify": "#D4FF8A",
        "fortify_spill": "#D4FF8A",
        "sabotage": "#B38CFF",
        "sabotage_spill": "#B38CFF",
        "rewire_cut": "#FF7AD1",
        "rewire_bridge": "#5DEBD3",
        "resupply": "#8A94A6",
        "player_success": "#45D483",
        "player_fail": "#FF5B79",
        "ai_success": "#45D483",
        "ai_fail": "#FF5B79",
        "event": "#FFCC6A",
    }

    node_x, node_y = [], []
    node_color, node_size, node_text, node_hover = [], [], [], []
    node_line_color, node_line_width, node_symbol = [], [], []

    for sector, (x, y) in pos.items():
        info = (sector_control or {}).get(sector, {})
        controller = info.get("controller", "Neutral")
        heat = int(info.get("heat", 0) or 0)
        stability = int(info.get("stability", 0) or 0)
        is_dead = stability <= 0
        act = activity.get(sector, {})
        act_cycle = int(act.get("cycle", -999))
        act_kind = str(act.get("kind", ""))
        is_active = (current_cycle - act_cycle) <= active_window

        node_x.append(x)
        node_y.append(y)
        base_color = palette.get(controller, palette["Neutral"])
        if is_dead:
            base_color = "#2A2F3A"
        node_color.append(base_color)
        size = 20 + min(24, max(0, heat)) * 0.24
        if is_active:
            size += 7
        node_size.append(size)
        node_text.append(sector.replace(" ", "<br>", 1) if " " in sector else sector)
        node_hover.append(
            f"<b>{sector}</b><br>"
            f"Controller: {controller}<br>"
            f"Heat: {heat}<br>"
            f"Stability: {stability}<br>"
            + (f"Recent activity: {act_kind} (cycle {act_cycle})<br>" if act_kind else "")
        )
        if is_dead:
            node_symbol.append("x")
            node_line_color.append("#7B8496")
            node_line_width.append(3)
        elif is_active and act_kind:
            node_symbol.append("circle")
            node_line_color.append(kind_outline.get(act_kind, "#E8ECF5"))
            node_line_width.append(4)
        else:
            node_symbol.append("circle")
            node_line_color.append("rgba(255,255,255,0.42)")
            node_line_width.append(2)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="rgba(228, 234, 248, 0.95)", family="Inter, system-ui, sans-serif"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            symbol=node_symbol,
            line=dict(width=node_line_width, color=node_line_color),
            opacity=0.96,
        ),
        showlegend=False,
    )

    legend_traces: list[go.Scatter] = []
    for name in sorted(palette.keys()):
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=True,
                marker=dict(
                    size=11,
                    color=palette[name],
                    line=dict(width=1, color="rgba(255,255,255,0.35)"),
                ),
            )
        )

    fig = go.Figure(data=[*legend_traces, edge_glow, edge_core, edge_hit, node_trace])
    fig.update_layout(
        uirevision="threadfall-ruin-topology",
        height=540,
        margin=dict(l=8, r=8, t=52, b=8),
        paper_bgcolor="#0b0d12",
        plot_bgcolor="#0b0d12",
        font=dict(color="#B4BDCF", family="Inter, system-ui, sans-serif", size=12),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, fixedrange=False, constrain="domain"),
        yaxis=dict(visible=False, fixedrange=False, constrain="domain"),
        dragmode="pan",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            x=0,
            xanchor="left",
            font=dict(size=10, color="#9AA6BC"),
            bgcolor="rgba(15,18,28,0.72)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#151a28", font_size=12, font_family="Inter, system-ui, sans-serif"),
    )
    return fig


def _sector_latlon() -> dict[str, tuple[float, float]]:
    """Hand-tuned lat/lon placement for a spherical topology view (degrees)."""
    return {
        "Archive Chamber Theta": (55.0, -55.0),
        "Mnemonic Scriptorium": (30.0, -15.0),
        "Signal Orchard": (10.0, -95.0),
        "Mirror Fault Atrium": (25.0, -150.0),
        "Inversion Causeway": (15.0, 40.0),
        "Vault Spine 3": (0.0, 95.0),
        "Thermal Lens Cathedral": (42.0, 120.0),
        "Obsidian Switchyard": (-10.0, 150.0),
        "Flooded Transit Gallery": (-30.0, 95.0),
        "Tide-Locked Rotunda": (-55.0, 40.0),
        "Resonance Stairwell": (-35.0, -30.0),
        "Null Pump Station": (-15.0, -120.0),
        "Gravimetric Choir": (5.0, 10.0),
    }


def _unit_from_latlon(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    v = np.array([x, y, z], dtype=float)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _great_circle_points(u: np.ndarray, v: np.ndarray, *, steps: int = 28) -> np.ndarray:
    """Return Nx3 points on the great-circle arc from u to v (unit vectors)."""
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    u = u / (float(np.linalg.norm(u)) or 1.0)
    v = v / (float(np.linalg.norm(v)) or 1.0)
    dot = float(np.clip(float(u @ v), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-6:
        pts = np.repeat(u.reshape(1, 3), repeats=max(2, int(steps)), axis=0)
        return pts
    sin_theta = math.sin(theta)
    ts = np.linspace(0.0, 1.0, num=max(2, int(steps)))
    pts = []
    for t in ts:
        a = math.sin((1.0 - float(t)) * theta) / sin_theta
        b = math.sin(float(t) * theta) / sin_theta
        w = a * u + b * v
        w = w / (float(np.linalg.norm(w)) or 1.0)
        pts.append(w)
    return np.array(pts, dtype=float)


def _build_topology_globe_figure(*, sector_control, factions) -> go.Figure:
    from threadfall.game import current_sector_graph

    controllers = {f["name"] for f in (factions or [])}
    palette = {
        "Neutral": "#8A94A6",
        "Helios Prospectors": "#FFCC6A",
        "Glass Archive": "#7EE7FF",
        "Morrow Division": "#B38CFF",
    }
    default_colors = ["#FFCC6A", "#7EE7FF", "#B38CFF", "#FF5B79", "#D4FF8A", "#DCCFFF"]
    for idx, name in enumerate(sorted(controllers)):
        palette.setdefault(name, default_colors[idx % len(default_colors)])

    latlon = _sector_latlon()
    unit = {s: _unit_from_latlon(*latlon[s]) for s in latlon}

    # Graticule (lat/lon grid) slightly under the node radius so it reads as a surface.
    grid_r = 0.985
    grid_traces: list[go.Scatter3d] = []
    for lat in [-60, -30, 0, 30, 60]:
        xs, ys, zs = [], [], []
        for lon in range(-180, 181, 6):
            p = _unit_from_latlon(lat, lon) * grid_r
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
        grid_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="rgba(180,200,255,0.12)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for lon in range(-180, 181, 30):
        xs, ys, zs = [], [], []
        for lat in range(-90, 91, 6):
            p = _unit_from_latlon(lat, lon) * grid_r
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
        grid_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="rgba(180,200,255,0.10)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Shared sector / cycle context (edges + nodes).
    activity = st.session_state.get("sector_activity", {}) or {}
    current_cycle = int(st.session_state.get("world_state", {}).get("cycle", 0))
    active_window = 1
    edge_activity_window = 2
    activities_recent: list[dict[str, Any]] = list(st.session_state.get("activities", [])[-40:])
    kind_outline = {
        "extract": "#FFCC6A",
        "extract_spill": "#FFCC6A",
        "research": "#7EE7FF",
        "research_spill": "#7EE7FF",
        "fortify": "#D4FF8A",
        "fortify_spill": "#D4FF8A",
        "sabotage": "#B38CFF",
        "sabotage_spill": "#B38CFF",
        "rewire_cut": "#FF7AD1",
        "rewire_bridge": "#5DEBD3",
        "resupply": "#8A94A6",
        "player_success": "#45D483",
        "player_fail": "#FF5B79",
        "ai_success": "#45D483",
        "ai_fail": "#FF5B79",
        "event": "#FFCC6A",
    }

    # Edge arcs on the sphere — color / width / hover from sector pulses + activity log (rewire, etc.).
    graph = current_sector_graph()
    seen = set()
    edge_traces: list[go.Scatter3d] = []
    for a, nbrs in graph.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            if a not in unit or b not in unit:
                continue
            pts = _great_circle_points(unit[a], unit[b], steps=30)
            ecolor, ewidth, ehover = _globe_style_corridor_edge(
                a,
                b,
                current_cycle=current_cycle,
                edge_window=edge_activity_window,
                sector_activity=activity,
                activities_recent=activities_recent,
                sector_control=sector_control or {},
                kind_hex=kind_outline,
            )
            edge_traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(color=ecolor, width=float(ewidth)),
                    hoverinfo="text",
                    hovertext=ehover,
                    showlegend=False,
                )
            )

    dead: dict[str, list] = {"xs": [], "ys": [], "zs": [], "sizes": [], "colors": [], "hover": []}
    active: dict[str, list] = {"xs": [], "ys": [], "zs": [], "sizes": [], "colors": [], "hover": [], "line": []}
    calm: dict[str, list] = {"xs": [], "ys": [], "zs": [], "sizes": [], "colors": [], "hover": []}
    labels_x, labels_y, labels_z, labels_text = [], [], [], []
    for sector, u in unit.items():
        info = (sector_control or {}).get(sector, {})
        controller = str(info.get("controller", "Neutral"))
        heat = int(info.get("heat", 0) or 0)
        stability = int(info.get("stability", 0) or 0)
        is_dead = stability <= 0
        act = activity.get(sector, {})
        act_cycle = int(act.get("cycle", -999))
        act_kind = str(act.get("kind", ""))
        is_active = (current_cycle - act_cycle) <= active_window

        base = palette.get(controller, palette["Neutral"])
        if is_dead:
            base = "#2A2F3A"
        size = 6 + min(22, max(0, heat)) * 0.10 + (3 if is_active else 0)
        hov = (
            f"<b>{sector}</b><br>"
            f"Controller: {controller}<br>"
            f"Heat: {heat}<br>"
            f"Stability: {stability}<br>"
            + (f"Recent activity: {act_kind} (cycle {act_cycle})<br>" if act_kind else "")
        )
        p = u * 1.01
        px, py, pz = float(p[0]), float(p[1]), float(p[2])

        if is_dead:
            bucket = dead
        elif is_active and act_kind:
            bucket = active
        else:
            bucket = calm
        bucket["xs"].append(px)
        bucket["ys"].append(py)
        bucket["zs"].append(pz)
        bucket["sizes"].append(size)
        bucket["colors"].append(base)
        bucket["hover"].append(hov)
        if bucket is active:
            bucket["line"].append(str(kind_outline.get(act_kind, "#E8ECF5")))

        lp = u * 1.06
        labels_x.append(float(lp[0]))
        labels_y.append(float(lp[1]))
        labels_z.append(float(lp[2]))
        labels_text.append(sector)

    node_traces: list[go.Scatter3d] = []
    if calm["xs"]:
        node_traces.append(
            go.Scatter3d(
                x=calm["xs"],
                y=calm["ys"],
                z=calm["zs"],
                mode="markers",
                marker=dict(
                    size=calm["sizes"],
                    color=calm["colors"],
                    line=dict(width=5, color="rgba(255,255,255,0.45)"),
                    opacity=0.98,
                ),
                hoverinfo="text",
                hovertext=calm["hover"],
                showlegend=False,
            )
        )
    if active["xs"]:
        # Scatter3d allows per-marker colors, but not per-marker line widths — split by outline color.
        by_line: dict[str, dict[str, list]] = {}
        for i, c in enumerate(active["line"]):
            by_line.setdefault(c, {"xs": [], "ys": [], "zs": [], "sizes": [], "colors": [], "hover": []})
            b = by_line[c]
            b["xs"].append(active["xs"][i])
            b["ys"].append(active["ys"][i])
            b["zs"].append(active["zs"][i])
            b["sizes"].append(active["sizes"][i])
            b["colors"].append(active["colors"][i])
            b["hover"].append(active["hover"][i])
        for c, b in by_line.items():
            node_traces.append(
                go.Scatter3d(
                    x=b["xs"],
                    y=b["ys"],
                    z=b["zs"],
                    mode="markers",
                    marker=dict(
                        size=b["sizes"],
                        color=b["colors"],
                        line=dict(width=7, color=c),
                        opacity=0.98,
                    ),
                    hoverinfo="text",
                    hovertext=b["hover"],
                    showlegend=False,
                )
            )
    if dead["xs"]:
        node_traces.append(
            go.Scatter3d(
                x=dead["xs"],
                y=dead["ys"],
                z=dead["zs"],
                mode="markers",
                marker=dict(
                    size=dead["sizes"],
                    color=dead["colors"],
                    line=dict(width=6, color="#7B8496"),
                    symbol="x",
                    opacity=0.98,
                ),
                hoverinfo="text",
                hovertext=dead["hover"],
                showlegend=False,
            )
        )
    label_trace = go.Scatter3d(
        x=labels_x,
        y=labels_y,
        z=labels_z,
        mode="text",
        text=labels_text,
        textfont=dict(size=10, color="rgba(230, 236, 248, 0.92)", family="Inter, system-ui, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
    )

    # Rival expedition units: diamond markers offset from their last resolved sector.
    by_sec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ag in st.session_state.get("ai_agents", []):
        name = str(ag.get("name", ""))
        if not name:
            continue
        sec = ag.get("last_sector") or _infer_rival_last_sector_from_history(name)
        if not isinstance(sec, str) or sec not in unit:
            continue
        by_sec[sec].append(ag)

    rival_colors = ["#FF7A45", "#5DEBD3", "#F472B6", "#A3E635", "#FACC15", "#38BDF8"]
    rx, ry, rz, rh, rcols = [], [], [], [], []
    gslot = 0
    for sector in sorted(by_sec.keys()):
        group = sorted(by_sec[sector], key=lambda a: str(a.get("name", "")))
        n_same = len(group)
        for slot, ag in enumerate(group):
            pos = _rival_unit_position(unit[sector], slot=slot, n_same=n_same)
            rx.append(float(pos[0]))
            ry.append(float(pos[1]))
            rz.append(float(pos[2]))
            intel = int(ag.get("intel", 0) or 0)
            arts = int(ag.get("artifacts", 0) or 0)
            outc = str(ag.get("last_outcome", "None"))
            mt = str(ag.get("last_mission_type", "standard"))
            cyc = ag.get("last_cycle")
            cyc_line = f"Cycle {cyc}<br>" if cyc is not None and str(cyc) != "" else ""
            rh.append(
                f"<b>Rival · {ag.get('name', '')}</b><br>Last op: {sector}<br>"
                f"Mission type: {mt}<br>Outcome: {outc}<br>"
                f"Intel {intel} · Artifacts {arts}<br>" + cyc_line
            )
            rcols.append(rival_colors[gslot % len(rival_colors)])
            gslot += 1

    rival_trace: go.Scatter3d | None = None
    if rx:
        rival_trace = go.Scatter3d(
            x=rx,
            y=ry,
            z=rz,
            mode="markers",
            marker=dict(
                size=11,
                color=rcols,
                symbol="diamond",
                line=dict(width=2, color="rgba(255,255,255,0.85)"),
                opacity=0.98,
            ),
            hoverinfo="text",
            hovertext=rh,
            name="Rival expeditions",
            showlegend=True,
        )

    data_layers: list = [*grid_traces, *edge_traces, *node_traces, label_trace]
    if rival_trace is not None:
        data_layers.append(rival_trace)
    fig = go.Figure(data=data_layers)
    fig.update_layout(
        uirevision="threadfall-ruin-globe",
        height=840,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#0b0d12",
        plot_bgcolor="#0b0d12",
        font=dict(color="#B4BDCF", family="Inter, system-ui, sans-serif", size=12),
        showlegend=rival_trace is not None,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0b0d12",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.9)),
        ),
    )
    return fig


def _render_topology_doctrine() -> None:
    """In-world brief on graph-based pressure and the rewire action (shown under the map)."""
    with st.expander("Tactical doctrine · corridors & rewire", expanded=False):
        st.markdown(
            """
**What the graph means**  
Only sectors connected by a **solid link** are neighbors. Extract, research, fortify, and sabotage
**spill** along those links—not pixel distance on the map.

**Sabotage staging**  
When a faction sabotages a **named rival**, it contests a sector in its own **preferred** set with the **shortest hop count**
(BFS) toward the sector where that **rival’s interest** is strongest. Topology decides where the blow lands.

**Rewire — `bridge|A|B`** (agent action)  
Adds a corridor. Factions use it to **shorten BFS** from their **preferred** sectors toward an enemy’s interest pocket (sharper sabotage staging),
or to **open a new spill fan** so extract / fortify / sabotage shock reaches different neighbors next cycle. Costs leverage; bridges add a little intel and heat.

**Rewire — `cut|A|B`**  
Removes a corridor. Cuts **lengthen or block** paths rivals use toward ground you care about, or **stop spill** along one edge.
Too many cuts can **fragment** the graph: if sectors disconnect, shortest-path picks get noisier—cut with intent.

**Timing**  
Agents rewire when **leverage** can absorb the cost and the **next cycles** can exploit the new links. Otherwise it is spent geometry.

**Crisis cadence**  
Blackout-style events cannot stack every cycle: the sim enforces **breathing room** after a crisis, dampens repeats from the same faction, and eases spawns when the **threat clock** is already high.
            """.strip()
        )


def render_topology_map(*, sector_control, factions, map_slot: str = "main") -> None:
    fp = _topology_fingerprint(sector_control=sector_control, factions=factions)
    cache = st.session_state.get("_topology_fig_cache")
    if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == fp:
        fig = cache[1]
    else:
        fig = _build_topology_figure(sector_control=sector_control, factions=factions)
        st.session_state["_topology_fig_cache"] = (fp, fig)

    with st.container(border=True):
        # Globe first so it is the default selected tab (Streamlit opens tab index 0).
        tabs = st.tabs(["Globe", "2D Map"])

        with tabs[0]:
            st.caption(
                "Globe topology: rotate the sphere (drag) · scroll to zoom · edges are great-circle arcs on the surface. "
                "Corridor **color and thickness** react to recent sector pulses, mission outcomes, faction actions (incl. rewire), and endpoint heat. "
                "Diamond markers: **rival expedition units** at their last resolved sector (offset when stacked)."
            )
            globe_cache = st.session_state.get("_topology_globe_fig_cache")
            if isinstance(globe_cache, tuple) and len(globe_cache) == 2 and globe_cache[0] == fp:
                globe_fig = globe_cache[1]
            else:
                globe_fig = _build_topology_globe_figure(sector_control=sector_control, factions=factions)
                st.session_state["_topology_globe_fig_cache"] = (fp, globe_fig)

            plotly_kwargs_globe: dict = dict(
                use_container_width=True,
                key=f"ruin_topology_globe_{map_slot}_{fp[:20]}",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "responsive": True,
                    "doubleClick": "reset",
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )
            try:
                st.plotly_chart(globe_fig, theme=None, **plotly_kwargs_globe)
            except TypeError:
                st.plotly_chart(globe_fig, **plotly_kwargs_globe)

        with tabs[1]:
            st.caption(
                "2D topology: pan/drag · scroll or pinch to zoom · hover nodes (status) or corridor midpoints (link summary)."
            )
            plotly_kwargs_2d: dict = dict(
                use_container_width=True,
                key=f"ruin_topology_map_{map_slot}_{fp[:20]}",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "responsive": True,
                    "doubleClick": "reset",
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )
            try:
                st.plotly_chart(fig, theme=None, **plotly_kwargs_2d)
            except TypeError:
                st.plotly_chart(fig, **plotly_kwargs_2d)

        _render_topology_doctrine()


def render_memory_results(results) -> None:
    if not results:
        st.caption("No memory results yet. Store a mission memory first, then query the archive.")
        return

    for result in results:
        with st.container(border=True):
            st.markdown(f"**Memory ID:** `{result.id}`")
            st.markdown(f"**Similarity:** `{result.score:.4f}`")
            st.write(result.text)
            payload = result.payload
            if payload:
                st.json(payload)


def render_world_state(world_state) -> None:
    cols = st.columns(4)
    cols[0].metric("Cycle", world_state["cycle"])
    cols[1].metric("Intel", world_state["intel"])
    cols[2].metric("Supplies", world_state["supplies"])
    cols[3].metric("Threat Clock", world_state["threat_clock"])
    st.progress(min(max(world_state["sector_stability"], 0), 100) / 100.0, text="Sector Stability")


def render_faction_state(factions) -> None:
    for faction in factions:
        with st.container(border=True):
            top = st.columns([1.2, 1, 1, 1])
            top[0].markdown(f"**{faction['name']}**")
            top[1].markdown(f"Stance: `{faction['stance']}`")
            top[2].markdown(f"Pressure: `{faction['pressure']}`")
            top[3].markdown(f"Leverage: `{faction['leverage']}`")
            st.markdown(
                _badge(faction["stance"], _severity_tone(faction["stance"]))
                + _badge(f"Interest: {faction['interest']}", "info"),
                unsafe_allow_html=True,
            )
            st.caption(f"Primary interest: {faction['interest']}")
            st.write(f"Goal: {faction['goal']}")
            st.caption(f"Rival: {faction['rival']} · Last outcome: {faction['last_outcome']}")
            recent = faction.get("recent_memory", [])
            if recent:
                for memory in recent[::-1]:
                    st.markdown(f"- {memory}")
            else:
                st.caption("No faction memory yet.")


def render_mas_panel() -> None:
    from threadfall.mas import MAS_DEFAULT_POLICY, init_mas, mas_state

    init_mas()
    mas = mas_state()
    st.subheader("MAS layer")
    st.caption(
        "Per-faction policies and turn order for autonomous cycles. "
        "Bandit arms learn from factor-weighted expedition deltas after each action."
    )
    order_opts = ("rotation", "shuffle", "fixed")
    cur_order = str(mas.get("order", "rotation"))
    order_idx = order_opts.index(cur_order) if cur_order in order_opts else 0
    mas["order"] = st.selectbox("Faction turn order", order_opts, index=order_idx, key="mas_global_order")
    if mas["order"] == "rotation":
        st.caption(f"Rotation offset (advances each full cycle): `{mas.get('rotation_idx', 0)}`")

    agents = mas.setdefault("agents", {})
    for faction in st.session_state.get("factions", []):
        name = str(faction.get("name", ""))
        if not name:
            continue
        slot = agents.setdefault(name, {})
        with st.expander(f"Agent · {name}", expanded=False):
            pol_opts = ["llm", "heuristic", "bandit"]
            cur_pol = str(slot.get("policy", MAS_DEFAULT_POLICY))
            pol_i = pol_opts.index(cur_pol) if cur_pol in pol_opts else pol_opts.index(MAS_DEFAULT_POLICY)
            slot["policy"] = st.selectbox(
                "Decision policy",
                pol_opts,
                index=pol_i,
                key=f"mas_policy_{name}",
                help="LLM: model JSON. Heuristic: rules engine. Bandit: ε-greedy over actions with learned values.",
            )
            slot["epsilon"] = st.slider(
                "Bandit exploration (ε)",
                min_value=0.0,
                max_value=0.5,
                value=float(slot.get("epsilon", 0.12)),
                step=0.01,
                key=f"mas_eps_{name}",
            )
            st.markdown("**Bandit reward weights** (effect deltas × weight, summed)")
            factors = slot.setdefault("factors", {})
            fcols = st.columns(3)
            keys = [
                ("intel", fcols[0]),
                ("leverage", fcols[0]),
                ("threat", fcols[1]),
                ("stability", fcols[1]),
                ("supplies", fcols[2]),
                ("pressure", fcols[2]),
            ]
            for key, col in keys:
                with col:
                    factors[key] = st.slider(
                        key,
                        min_value=-2.0,
                        max_value=2.0,
                        value=float(factors.get(key, 0.0)),
                        step=0.05,
                        key=f"mas_f_{name}_{key}",
                    )
            bandit = slot.get("bandit", {})
            n_obs = len(bandit.get("values", {}))
            st.caption(f"Bandit states observed: {n_obs}")


def render_inventory(inventory) -> None:
    if not inventory:
        st.caption("No artifacts recovered yet.")
        return

    for item in inventory[-8:][::-1]:
        with st.container(border=True):
            st.markdown(f"**{item['name']}**")
            st.caption(f"{item['rarity']} artifact from {item['source_sector']}")
            st.write(item["effect"])


def render_history(history) -> None:
    if not history:
        st.caption("No expedition history yet.")
        return

    for item in history[-8:][::-1]:
        with st.container(border=True):
            st.markdown(
                f"**Cycle {item['cycle']}** · `{item['outcome']}` · {item['sector']} · {item['objective']}"
            )
            st.markdown(
                _badge(item["outcome"], _severity_tone(item["outcome"]))
                + _badge(item["sector"], "info"),
                unsafe_allow_html=True,
            )
            st.caption(f"Sponsor: {item['sponsor']} · Rival: {item.get('rival', 'Unknown')}")
            st.write(item["summary"])


def render_command_bar(services, *, ended: bool) -> None:
    from threadfall.game import (
        append_log,
        autoplay_state,
        build_mission,
        configure_autoplay,
        game_status,
        remember_current_mission,
        reset_campaign,
        resolve_current_mission,
        run_autonomous_stack_tick,
        run_ml_swarm_until_endgame,
        stop_autoplay,
        undo_last_tick,
    )

    with st.sidebar:
        st.markdown("### Command Bar")
        swarm_toast = st.session_state.pop("_cmd_swarm_toast", None)
        if swarm_toast:
            st.success(swarm_toast)
        status = game_status()
        st.caption(f"Campaign: `{status['state']}`")
        if ended:
            st.warning("Campaign ended. Reset to unlock simulation controls.")
            if st.button("Reset Campaign", use_container_width=True, key="cmd_reset_campaign"):
                reset_campaign()
                st.rerun()

        st.markdown("#### Mission")
        if st.button("Generate New Expedition", use_container_width=True, disabled=ended, key="cmd_generate"):
            st.session_state["current_mission"] = build_mission(
                squad_name=st.session_state.get("squad_name", "Orpheus Unit")
            )
            append_log("Generated a fresh expedition profile.")
            st.rerun()

        if st.button("Store Mission Memory", use_container_width=True, disabled=ended, key="cmd_store"):
            mission = st.session_state["current_mission"]
            memory_id = remember_current_mission(services.memory, mission)
            append_log(f"Stored expedition memory: {memory_id}")
            st.rerun()

        if st.button(
            "Resolve Expedition",
            type="primary",
            use_container_width=True,
            disabled=ended,
            key="cmd_resolve",
        ):
            result = resolve_current_mission(services.memory)
            append_log(f"Resolved expedition: {result['outcome']} (cycle {result['cycle']}).")
            st.rerun()

        st.markdown("#### Simulation")
        run_cols = st.columns(2)
        if run_cols[0].button("Run 1 Cycle", use_container_width=True, disabled=ended, key="cmd_run_1"):
            tick = run_autonomous_stack_tick(services.memory, services.decision_engine)
            append_log(
                f"Stack tick: {len(tick.get('player', []))} player expedition(s), "
                f"{len(tick.get('factions', []))} faction decision(s)."
            )
            st.rerun()
        if run_cols[1].button("Run 5 Cycles", use_container_width=True, disabled=ended, key="cmd_run_5"):
            tp = tf = 0
            for _ in range(5):
                tick = run_autonomous_stack_tick(services.memory, services.decision_engine)
                tp += len(tick.get("player", []))
                tf += len(tick.get("factions", []))
                if st.session_state["endgame"]["ended"]:
                    break
            append_log(f"Stack x5: {tp} player expedition(s), {tf} faction decision(s).")
            st.rerun()

        with st.expander("ML swarm (end-to-end)", expanded=False):
            st.caption(
                "Run stack ticks until endgame or cap. Restores stack/MAS after if enabled. "
                "No encounter UI on player autoplay."
            )
            cmd_swarm_cap = int(
                st.number_input(
                    "Max ticks",
                    min_value=10,
                    max_value=8000,
                    value=2000,
                    step=50,
                    key="cmd_swarm_cap",
                )
            )
            cmd_swarm_throughput = st.checkbox(
                "Max throughput (3× player, 8× faction / tick)",
                value=True,
                key="cmd_swarm_throughput",
            )
            cmd_swarm_bandit = st.checkbox(
                "All factions → bandit (fast, no faction LLM)",
                value=False,
                key="cmd_swarm_bandit",
            )
            cmd_swarm_restore = st.checkbox(
                "Restore stack + MAS after",
                value=True,
                key="cmd_swarm_restore",
            )
            if st.button(
                "Run ML swarm to endgame",
                use_container_width=True,
                disabled=ended,
                key="cmd_swarm_run",
            ):
                with st.spinner("Swarm…"):
                    out = run_ml_swarm_until_endgame(
                        services.memory,
                        services.decision_engine,
                        max_stack_ticks=cmd_swarm_cap,
                        max_throughput=cmd_swarm_throughput,
                        mas_bandit_only=cmd_swarm_bandit,
                        restore_stack_after=cmd_swarm_restore,
                        restore_mas_after=cmd_swarm_restore,
                    )
                _tail = (
                    "Hit tick cap while Active."
                    if out["hit_cap"]
                    else f"Ended: {out.get('result') or 'Active'}."
                )
                st.session_state["_cmd_swarm_toast"] = (
                    f"Swarm: {out['ticks']} tick(s), {out['player_resolutions']} player, "
                    f"{out['faction_rows']} faction rows. {_tail}"
                )
                st.rerun()

        st.markdown("#### Autoplay")
        autoplay = autoplay_state()
        ticks = st.number_input("Ticks", min_value=1, max_value=50, value=5, step=1, key="cmd_ticks")
        rounds_per_tick = st.number_input(
            "Rounds/tick", min_value=1, max_value=5, value=1, step=1, key="cmd_rounds_per_tick"
        )
        delay_seconds = st.slider(
            "Delay (s)", min_value=0.2, max_value=3.0, value=0.8, step=0.1, key="cmd_delay_seconds"
        )
        start_col, stop_col = st.columns(2)
        if start_col.button("Start", use_container_width=True, disabled=ended, key="cmd_autoplay_start"):
            configure_autoplay(True, int(ticks), int(rounds_per_tick), float(delay_seconds))
            append_log("Autoplay armed.")
            st.rerun()
        if stop_col.button("Stop", use_container_width=True, key="cmd_autoplay_stop"):
            stop_autoplay()
            append_log("Autoplay stopped.")
            st.rerun()
        if autoplay["enabled"]:
            st.caption(
                f"Autoplay running · remaining `{autoplay['remaining_ticks']}` · rounds/tick `{autoplay['rounds_per_tick']}`"
            )

        st.markdown("#### Undo")
        can_undo = bool(st.session_state.get("_undo_stack"))
        if st.button("Undo last tick", use_container_width=True, disabled=not can_undo, key="cmd_undo"):
            ok, msg = undo_last_tick()
            append_log(msg)
            st.rerun()
        if not can_undo:
            st.caption("Undo will appear after you resolve a mission or run a simulation cycle.")

        st.divider()
        st.markdown("#### Runtime")
        st.caption(f"Embedding device: `{services.embedder.device.upper()}`")
        st.caption(f"Memory backend: `{services.memory.backend_name}`")
        st.caption(f"Decision engine: `{services.decision_engine.backend_name}`")


def render_activity_feed(activities) -> None:
    if not activities:
        st.caption("No autonomous agent activity yet.")
        return

    for item in activities[-12:][::-1]:
        with st.container(border=True):
            st.markdown(
                f"**Cycle {item['cycle']}** · `{item['faction']}` · `{item['action']}` → `{item['target']}`"
            )
            st.markdown(
                _badge(item["action"].upper(), _severity_tone(item["decision_status"] if item.get("decision_status") in {"Fallback"} else "Active"))
                + _badge(item["target"], "info")
                + _badge(item.get("decision_source", "heuristic"), "accent"),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Confidence: {item['confidence']} · Source: {item.get('decision_source', 'heuristic')} · Status: {item.get('decision_status', 'n/a')}"
            )
            if item.get("negotiation_move"):
                st.markdown(
                    _badge(f"Negotiation: {item['negotiation_move']}", "warning")
                    + _badge(item.get("negotiation_partner", "unknown"), "neutral"),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Negotiation before action: {item['negotiation_move']} with {item.get('negotiation_partner', 'unknown')}"
                )
            st.write(item["reasoning"])
            st.markdown("**What The Agent Decided**")
            st.write(
                f"**{item['faction']}** chose to **{item['action']}** and aimed that move at **{item['target']}**. "
                f"This is the faction's main operational choice for that cycle."
            )
            st.markdown("**Why This Matters**")
            if item["action"] == "extract":
                st.write(
                    "This means the faction is trying to gain resources, relic access, or long-term influence from a sector."
                )
            elif item["action"] == "research":
                st.write(
                    "This means the faction is investing in knowledge and stability, which can improve future decisions and lower uncertainty."
                )
            elif item["action"] == "sabotage":
                st.write(
                    "This means the faction is trying to disrupt a rival rather than improve its own position directly."
                )
            elif item["action"] == "fortify":
                st.write(
                    "This means the faction is trying to lock down control and make a sector harder for others to challenge."
                )
            elif item["action"] == "rewire":
                st.write(
                    "This means the faction is altering which sectors count as adjacent on the topology graph—changing "
                    "how spillover, sabotage routing, and pressure propagate for later cycles."
                )
            else:
                st.write(
                    "This means the faction is pausing expansion to rebuild supplies and recover operational capacity."
                )
            effects = item["effects"]
            _render_delta_badges(
                [
                    ("intel", effects["intel_delta"]),
                    ("supplies", effects["supplies_delta"]),
                    ("threat", effects["threat_delta"]),
                    ("stability", effects["stability_delta"]),
                    ("pressure", effects["pressure_delta"]),
                    ("leverage", effects["leverage_delta"]),
                ]
            )
            st.markdown("**Effect Summary**")
            st.write(
                f"This move changed the global simulation by shifting intel, supplies, threat, stability, pressure, or leverage. "
                f"Positive numbers mean an increase. Negative numbers mean a decrease."
            )
            refs = item.get("memory_refs", [])
            if refs:
                st.caption(f"Retrieved memories: {', '.join(refs)}")
                st.write(
                    "These memory references show past records the agent looked at before deciding. "
                    "They are evidence the faction used for retrieval-augmented reasoning."
                )


def render_agent_summary(factions) -> None:
    table = []
    for faction in factions:
        recent = faction.get("recent_memory", [])
        table.append(
            {
                "Faction": faction["name"],
                "Stance": faction["stance"],
                "Pressure": faction["pressure"],
                "Leverage": faction["leverage"],
                "Last outcome": faction["last_outcome"],
                "Recent memory": recent[-1] if recent else "None",
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_rival_units(ai_agents) -> None:
    if not ai_agents:
        st.caption("No rival units configured. Set them in Overview to add competitors.")
        return

    table = []
    for agent in ai_agents:
        table.append(
            {
                "Unit": agent.get("name", "Rival Unit"),
                "Intel": int(agent.get("intel", 0)),
                "Artifacts": int(agent.get("artifacts", 0)),
                "Last outcome": agent.get("last_outcome", "None"),
                "Last mission": agent.get("last_mission_type", "standard"),
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_ai_expedition_feed(history) -> None:
    items = [h for h in (history or []) if h.get("type") == "ai_expedition"]
    if not items:
        st.caption("No rival expeditions recorded yet. Run agent cycles to advance rival units.")
        return

    for item in items[-10:][::-1]:
        with st.container(border=True):
            st.markdown(
                f"**Cycle {item['cycle']}** · `{item.get('actor', 'Rival Unit')}` · `{item['outcome']}` · {item['sector']} · {item['objective']}"
            )
            st.markdown(
                _badge(item["outcome"], _severity_tone(item["outcome"]))
                + _badge(item.get("mission_type", "standard"), "accent")
                + _badge(item["sector"], "info"),
                unsafe_allow_html=True,
            )
            if item.get("artifact_found"):
                st.caption("Artifact recovered.")
            st.caption(f"Sponsor: {item.get('sponsor', 'Unknown')} · Rival: {item.get('rival', 'Unknown')}")
            st.write(item.get("summary", ""))


def render_decision_engine_status(decision_engine) -> None:
    st.metric("Decision Backend", "OLLAMA" if decision_engine.is_available() else "FALLBACK")
    st.markdown(
        _badge("OLLAMA" if decision_engine.is_available() else "FALLBACK", "success" if decision_engine.is_available() else "warning"),
        unsafe_allow_html=True,
    )
    st.caption(f"Model: {decision_engine.model}")
    st.caption(f"Status: {decision_engine.status['detail']}")


def render_autoplay_status(state) -> None:
    mode = "RUNNING" if state["enabled"] and state["remaining_ticks"] > 0 else "IDLE"
    st.metric("Autoplay", mode)
    st.markdown(_badge(mode, "success" if mode == "RUNNING" else "neutral"), unsafe_allow_html=True)
    st.caption(
        f"Ticks remaining: {state['remaining_ticks']} · Rounds per tick: {state['rounds_per_tick']} · Delay: {state['delay_seconds']}s"
    )


def render_sector_control(sector_control) -> None:
    rows = []
    for sector, info in sector_control.items():
        top_interest = max(info["interest"], key=info["interest"].get)
        rows.append(
            {
                "Sector": sector,
                "Controller": info["controller"],
                "Heat": info["heat"],
                "Stability": info["stability"],
                "Top interest": f"{top_interest} ({info['interest'][top_interest]})",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_events(events) -> None:
    if not events:
        st.caption("No active world events.")
        return

    for event in events[-8:][::-1]:
        source = event.get("source_faction", "Unknown")
        target = event.get("target_faction", "Unknown")
        name = event.get("name", "Event")
        sector = event.get("sector", "Unknown sector")
        severity = event.get("severity", "Low")

        is_faction_escalation = (
            isinstance(source, str)
            and isinstance(target, str)
            and source not in {"The Ruin", "Unknown"}
            and target not in {"The Ruin", "Unknown"}
            and source != target
        )
        is_player_sourced = isinstance(source, str) and ("Unit" in source or "Squad" in source)
        is_ruin_sourced = source == "The Ruin" or target == "The Ruin"

        with st.container(border=True):
            st.markdown(
                f"**Cycle {event['cycle']}** · `{name}` · `{severity}` · {sector}"
            )
            st.markdown(
                _badge(severity, _severity_tone(severity))
                + _badge(sector, "info")
                + _badge(name, "accent"),
                unsafe_allow_html=True,
            )
            st.caption(f"Source: {source} · Target: {target}")
            st.write(event["summary"])
            st.markdown("**What This Means**")
            if is_faction_escalation:
                st.write(
                    f"**{source}** has created a new pressure point in **{sector}** that impacts **{target}**. "
                    "If left unchecked, this can shift sector control, raise counter-mission risk, and change the next mission offer."
                )
            elif is_ruin_sourced:
                st.write(
                    f"The ruin’s conditions shifted in **{sector}**. "
                    "Treat this as an environmental change that alters risk, stability, or the pace of escalation."
                )
            elif is_player_sourced:
                st.write(
                    f"This is a squad-side incident tied to operations in **{sector}**. "
                    "It represents a concrete cost (lost supplies, intel, or momentum) rather than a faction escalation."
                )
            else:
                st.write(
                    f"An event was recorded in **{sector}**. "
                    "Use the details below to understand how it changes risk, resources, or the next mission offer."
                )
            st.markdown("**Why It Happened**")
            st.write(event.get("why_now", "A faction escalation triggered this event."))
            st.markdown("**Likely Player Impact**")
            st.write(event.get("player_impact", "This event may influence the next mission offer."))
            st.markdown("**Recommended Interpretation**")
            if is_faction_escalation:
                st.write(
                    f"Treat this as a contested-maps signal: **{source}** is pushing an advantage in **{sector}**. "
                    "If you ignore it, you’re effectively allowing the political situation to drift without intervention."
                )
            elif is_ruin_sourced:
                st.write(
                    "Treat this as a clock tick from the environment. "
                    "If you want stability, favor containment/survey decisions and avoid high-heat actions."
                )
            elif is_player_sourced:
                st.write(
                    "Treat this as actionable operational feedback: adjust the next mission type, commit supplies, or pick a safer sector."
                )
            else:
                st.write("Treat this as context for the next cycle and adjust strategy accordingly.")


def render_counter_missions(counter_missions) -> None:
    if not counter_missions:
        st.caption("No queued counter-missions.")
        return

    for mission in counter_missions[:5]:
        with st.container(border=True):
            st.markdown(
                f"**{mission['sector']}** · `{mission['objective']}` · `{mission['difficulty']}`"
            )
            st.markdown(
                _badge(mission["difficulty"], _severity_tone(mission["difficulty"]))
                + _badge(mission["sector"], "info")
                + _badge("Counter Mission", "warning"),
                unsafe_allow_html=True,
            )
            st.caption(f"Sponsor: {mission['sponsor']} · Rival: {mission['rival']}")
            st.write(mission["summary"])
            st.markdown("**What This Means**")
            st.write(
                f"This is an urgent player-facing mission generated by faction conflict. "
                f"**{mission['sponsor']}** wants a response in **{mission['sector']}** because **{mission['rival']}** has shifted the balance there."
            )
            st.markdown("**Why It Is Queued**")
            st.write(
                "Counter-missions appear when the autonomous simulation creates a crisis that should override normal exploration. "
                "If one of these is queued, the next expedition is more likely to be reactive rather than routine."
            )
            st.markdown("**How To Read It**")
            st.write(
                "Think of this as the game telling you that the political or territorial situation changed fast enough to demand immediate intervention."
            )


def render_negotiation_feed(negotiations) -> None:
    if not negotiations:
        st.caption("No negotiation activity yet.")
        return

    for item in negotiations[-10:][::-1]:
        with st.container(border=True):
            st.markdown(
                f"**Cycle {item['cycle']}** · `{item['source']}` → `{item['partner']}` · `{item['move']}`"
            )
            st.markdown(
                _badge(item["move"].upper(), "warning")
                + _badge(item["partner"], "info")
                + _badge(item.get("source_model", "heuristic"), "accent"),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Confidence: {item['confidence']} · Source: {item.get('source_model', 'heuristic')} · Status: {item.get('status', 'n/a')}"
            )
            st.write(item["terms"])
            st.markdown("**What Happened**")
            st.write(
                f"Before taking an action, **{item['source']}** tried to influence **{item['partner']}** using **{item['move']}**."
            )
            st.markdown("**How To Interpret The Move**")
            if item["move"] == "threaten":
                st.write(
                    "This means the source faction is applying pressure and trying to force compliance through intimidation."
                )
            elif item["move"] == "bargain":
                st.write(
                    "This means the source faction is trying to trade short-term concessions for stability, time, or resources."
                )
            elif item["move"] == "align":
                st.write(
                    "This means the source faction is seeking temporary cooperation or shared benefit."
                )
            else:
                st.write(
                    "This means the source faction is trying to manipulate the other side through misleading or strategic framing."
                )
            effects = item["effects"]
            _render_delta_badges(
                [
                    ("src pressure", effects["source_pressure"]),
                    ("src leverage", effects["source_leverage"]),
                    ("partner pressure", effects["partner_pressure"]),
                    ("partner leverage", effects["partner_leverage"]),
                    ("supplies", effects["supplies_delta"]),
                    ("intel", effects["intel_delta"]),
                ]
            )
            st.markdown("**Effect Summary**")
            st.write(
                "These numbers show how the negotiation changed the relationship and the world state before the main action phase began."
            )


def render_endgame_summary(endgame, history, factions, sector_control, inventory) -> None:
    st.metric("Result", endgame["result"])
    st.metric("Winner", endgame["winner"])
    st.markdown(
        _badge(endgame["result"], _severity_tone(endgame["result"]))
        + _badge(endgame["winner"], "accent"),
        unsafe_allow_html=True,
    )
    st.caption(f"Triggered at cycle {endgame['trigger_cycle']}")
    st.write(endgame["summary"])

    st.subheader("Executive Summary")
    world = st.session_state.get("world_state", {})
    ai_agents = st.session_state.get("ai_agents", [])
    result = endgame.get("result", "Active")

    final_intel = int(world.get("intel", 0))
    final_supplies = int(world.get("supplies", 0))
    final_threat = int(world.get("threat_clock", 0))
    final_stability = int(world.get("sector_stability", 0))
    artifacts = len(inventory or [])
    cycle = int(world.get("cycle", endgame.get("trigger_cycle", 0) or 0))

    # Sector leader for interpretation.
    counts: dict[str, int] = {}
    for _, info in (sector_control or {}).items():
        controller = info.get("controller", "Neutral")
        counts[controller] = counts.get(controller, 0) + 1
    top_controller = max(counts, key=counts.get) if counts else "Neutral"

    bullets: list[str] = []
    bullets.append(f"Final cycle: **{cycle}** · Intel: **{final_intel}** · Artifacts: **{artifacts}**.")
    bullets.append(
        f"Ruin pressure ended at Threat Clock **{final_threat}** and Sector Stability **{final_stability}** "
        f"with Supplies **{final_supplies}** remaining."
    )
    if top_controller and top_controller != "Neutral":
        bullets.append(f"Territory leader at the end: **{top_controller}** controlled **{counts.get(top_controller, 0)}** sectors.")

    if result == "Player Victory":
        bullets.append(
            "You won because you accumulated enough **intel + artifacts** to secure dominance before the ruin or rivals could end the run."
        )
    elif result == "Rival Victory":
        rival = next((a for a in ai_agents if a.get("name") == endgame.get("winner")), None)
        if rival:
            bullets.append(
                f"You lost because **{rival.get('name')}** reached breakthrough first "
                f"(intel **{rival.get('intel', 0)}**, artifacts **{rival.get('artifacts', 0)}**) and locked the ruin."
            )
        else:
            bullets.append(
                "You lost because a rival unit reached breakthrough first and locked the ruin before you could match their progress."
            )
    elif result == "Player Defeat":
        if final_threat >= 100 or final_stability <= 0:
            bullets.append(
                "You lost because the ruin hit a terminal state (Threat Clock maxed or Stability collapsed) before you could stabilize the campaign."
            )
        elif final_supplies <= 0:
            bullets.append("You lost because the expedition hub ran out of supplies while crises piled up.")
        else:
            bullets.append("You lost due to campaign failure conditions being met before a decisive breakthrough.")
    elif result == "Faction Victory":
        bullets.append(
            "You lost because a faction converted sector control + leverage into dominance faster than you could disrupt it."
        )
    elif result == "Stalemate":
        bullets.append(
            "The campaign ended in stalemate: the run dragged on without a decisive breakthrough, and territorial control hardened into a long containment war."
        )
    else:
        bullets.append("The campaign ended due to an endgame condition being met.")

    # Tactical hints (post-mortem).
    if result in {"Player Defeat", "Faction Victory", "Rival Victory", "Stalemate"}:
        hints: list[str] = []
        if final_threat >= 70:
            hints.append("Run more **containment** (and avoid repeated high-heat extraction) to keep Threat Clock under control.")
        if final_supplies <= 2:
            hints.append("Protect supplies: commit fewer extras and interleave safer missions before long streaks of deep/heat-heavy runs.")
        if final_stability <= 25:
            hints.append("Prioritize stability: containment + survey loops and lower-heat sector choices reduce collapse risk.")
        if ai_agents:
            leader = max(ai_agents, key=lambda a: int(a.get("intel", 0)) + 20 * int(a.get("artifacts", 0)))
            hints.append(f"Watch rival tempo: **{leader.get('name')}** was the pace-setter—disrupt with lower-risk progress and earlier breakthroughs.")
        if hints:
            bullets.append("Post-mortem actions to try next run:")
            bullets.extend([f"- {h}" for h in hints[:4]])

    for line in bullets:
        st.markdown(f"- {line}")

    if endgame.get("details"):
        for line in endgame["details"]:
            st.markdown(f"- {line}")

    st.subheader("Final Sector Control")
    render_sector_control(sector_control)

    st.subheader("Final Faction State")
    render_agent_summary(factions)

    st.subheader("Recovered Artifacts")
    render_inventory(inventory)

    st.subheader("Recent Campaign History")
    render_history(history)


def render_timeline_charts(snapshots) -> None:
    if not snapshots or len(snapshots) < 2:
        st.caption("Not enough timeline data yet. Run a few mission resolutions or agent cycles first.")
        return

    df = pd.DataFrame(snapshots)
    st.markdown("**World Trend**")
    st.caption(
        "Tracks the global campaign state per cycle. "
        "`threat_clock` rising means the ruin is escalating toward failure; falling means containment/progress. "
        "`intel` rising is overall progress; `supplies` falling constrains actions; `sector_stability` falling signals collapse risk."
    )
    world_chart = df[["cycle", "threat_clock", "intel", "supplies", "sector_stability"]].set_index("cycle")
    st.line_chart(world_chart, use_container_width=True, height=240)

    st.markdown("**Faction Leverage Over Time**")
    st.caption(
        "Shows each faction’s political/momentum score. "
        "Upward trends mean that faction is winning negotiations/operations; downward trends mean setbacks. "
        "A single faction staying high for many cycles usually predicts a faction victory unless you intervene."
    )
    leverage_chart = df[
        [
            "cycle",
            "Helios Prospectors_leverage",
            "Glass Archive_leverage",
            "Morrow Division_leverage",
        ]
    ].set_index("cycle")
    st.line_chart(leverage_chart, use_container_width=True, height=240)

    st.markdown("**Sector Control Over Time**")
    st.caption(
        "Area chart of how many sectors each side controls each cycle (including neutral). "
        "When one faction’s area steadily grows, it is consolidating territory. "
        "Large swings often follow high-heat actions, counter-missions, or repeated successes in the same sector."
    )
    control_chart = df[
        [
            "cycle",
            "Helios Prospectors_control",
            "Glass Archive_control",
            "Morrow Division_control",
            "neutral_control",
        ]
    ].set_index("cycle")
    st.area_chart(control_chart, use_container_width=True, height=240)
