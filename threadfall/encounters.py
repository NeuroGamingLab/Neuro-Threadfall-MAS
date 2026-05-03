from __future__ import annotations

import base64
import json
from typing import Any


def _b64url_encode(data: dict[str, Any]) -> str:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(token: str) -> dict[str, Any]:
    padding = "=" * ((4 - len(token) % 4) % 4)
    raw = base64.urlsafe_b64decode((token + padding).encode("ascii"))
    return json.loads(raw.decode("utf-8"))


def decode_encounter_query(token: str) -> dict[str, Any]:
    return _b64url_decode(token)


def robotron_encounter_html(*, mission: dict[str, Any], context: dict[str, Any]) -> str:
    payload = {
        "version": 1,
        "mode": "arena",
        "mission_id": mission.get("mission_id", ""),
        "sector": mission.get("sector", ""),
        "difficulty": mission.get("difficulty", "Low"),
        "mission_type": mission.get("mission_type", "standard"),
        "sector_tags": list(mission.get("sector_tags", [])),
        "anomaly_tags": list(mission.get("anomaly_tags", [])),
        "world": {
            "threat_clock": int(context.get("threat_clock", 0)),
            "supplies": int(context.get("supplies", 0)),
            "sector_stability": int(context.get("sector_stability", 0)),
        },
        "sector_state": {
            "heat": int(context.get("heat", 0)),
            "stability": int(context.get("sector_local_stability", 0)),
        },
    }
    start = _b64url_encode(payload)
    title = f"Arena Encounter — {mission.get('sector', 'Unknown')}"
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; background:#0b1020; color:#e6edf3; font-family: ui-sans-serif, system-ui; }}
    .wrap {{ padding: 12px 14px; }}
    .hud {{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; }}
    .pill {{ padding:6px 10px; border-radius:999px; background:#162044; border:1px solid #2a3a74; font-weight:600; font-size:12px; }}
    .pill.bad {{ background:#3b1020; border-color:#7c2740; }}
    .pill.good {{ background:#123b2c; border-color:#1f6b4a; }}
    canvas {{ display:block; margin-top:12px; width:100%; max-width:980px; height:520px; border-radius:14px; border:1px solid #2a3a74; background: radial-gradient(ellipse at 50% 30%, #101a3d 0%, #070a15 70%); }}
    .hint {{ opacity:0.85; font-size:12px; margin-top:8px; }}
    .btn {{ margin-top:10px; padding:10px 12px; border-radius:12px; background:#2d4cff; border:none; color:white; font-weight:700; cursor:pointer; }}
    .btn:disabled {{ opacity:0.5; cursor:not-allowed; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hud">
      <div class="pill">MODE: ARENA</div>
      <div class="pill">MISSION: {mission.get("mission_type","standard")}</div>
      <div class="pill">DIFFICULTY: {mission.get("difficulty","Low")}</div>
      <div class="pill" id="t">TIME: 45.0</div>
      <div class="pill" id="s">SCORE: 0</div>
      <div class="pill" id="hp">HULL: 3</div>
      <div class="pill" id="combo">COMBO: x1.0</div>
      <div class="pill" id="grade">GRADE: —</div>
    </div>
    <canvas id="c" width="980" height="520"></canvas>
    <div class="hint">Controls: WASD/Arrow keys move · Space shoots · Survive the timer. Higher score increases chance of Success.</div>
    <button class="btn" id="finish" disabled>Return result to Threadfall</button>
  </div>
  <script>
    const START = "{start}";
    const state = JSON.parse(atob(START.replace(/-/g,'+').replace(/_/g,'/').padEnd(Math.ceil(START.length/4)*4,'=')));
    const missionId = state.mission_id || "";
    const diff = state.difficulty || "Low";
    const heat = (state.sector_state && typeof state.sector_state.heat === "number") ? state.sector_state.heat : 0;
    const threatClock = (state.world && typeof state.world.threat_clock === "number") ? state.world.threat_clock : 0;
    const missionType = state.mission_type || "standard";
    const difficultyMul = diff === "Terminal" ? 1.45 : diff === "Severe" ? 1.25 : diff === "Elevated" ? 1.1 : 1.0;
    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    const keys = new Set();
    let t = 45.0;
    // World pressure makes the encounter tighter.
    t -= Math.min(10, Math.floor(heat/12));
    t -= (threatClock >= 70 ? 4 : threatClock >= 40 ? 2 : 0);
    let score = 0;
    let hp = 3;
    let done = false;
    const player = {{ x: canvas.width/2, y: canvas.height/2, vx:0, vy:0 }};
    const bullets = [];
    const enemyBullets = [];
    const enemies = [];
    let spawn = 0;
    let combo = 1.0;
    let comboTimer = 0;
    let shake = 0;
    let hitFlash = 0;
    let lastKill = 0;

    // tiny audio helper (no assets)
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = AudioCtx ? new AudioCtx() : null;
    function beep(freq, dur, type="sine", gain=0.06) {{
      if (!audioCtx) return;
      const t0 = audioCtx.currentTime;
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = type;
      o.frequency.setValueAtTime(freq, t0);
      g.gain.setValueAtTime(gain, t0);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(t0); o.stop(t0 + dur);
    }}

    function rand(a,b){{ return a + Math.random()*(b-a); }}
    function clamp(v,a,b){{ return Math.max(a, Math.min(b,v)); }}
    function fire() {{
      bullets.push({{ x: player.x, y: player.y, vx: 520, vy: 0, life: 0.7 }});
      beep(520, 0.04, "square", 0.03);
    }}
    function spawnEnemy() {{
      const side = Math.floor(Math.random()*4);
      const x = side===0? -20 : side===1? canvas.width+20 : rand(0, canvas.width);
      const y = side===2? -20 : side===3? canvas.height+20 : rand(0, canvas.height);
      const roll = Math.random();
      let kind = "chaser";
      if (roll < 0.18) kind = "turret";
      else if (roll < 0.46) kind = "flanker";
      const baseSp = (60+rand(0,80))*difficultyMul;
      enemies.push({{
        kind,
        x, y,
        r: kind==="turret" ? 14 : 10+rand(0,6),
        sp: kind==="turret" ? baseSp*0.55 : baseSp,
        cd: rand(0.3, 0.9),
        angle: rand(0, Math.PI*2),
      }});
    }}
    function endEncounter() {{
      done = true;
      document.getElementById('finish').disabled = false;
      const target = (diff==="Low"? 55: diff==="Elevated"? 75: diff==="Severe"? 95: 120) + Math.floor(heat/10);
      const success = (hp > 0) && (score >= target);
      const bonusIntel = Math.round(Math.min(10, score/12));
      // Mission-type shaping: extraction runs have higher artifact odds if you meet the bar.
      const artifactBar = (missionType === "extraction") ? Math.floor(target*0.92) : 90;
      const artifactBonus = (score >= artifactBar) ? 1 : 0;
      const grade = score >= target*1.35 ? "S" : score >= target*1.15 ? "A" : score >= target ? "B" : score >= target*0.75 ? "C" : "D";
      // Encounter contract: explicit deltas + stage results.
      const intel_delta = bonusIntel;
      const supplies_delta = 0;
      const heat_delta = Math.max(0, Math.min(12, Math.floor(score/18)));
      const stability_delta = (success ? Math.min(6, Math.floor(score/35)) : -Math.min(6, 2 + Math.floor((target-score)/30)));
      const threat_delta = (success ? -Math.min(6, 1 + Math.floor(score/40)) : Math.min(10, 3 + Math.floor((target-score)/18)));
      const stage_results = [
        {{ stage: "Approach", success: hp >= 3, chance: 1.0 }},
        {{ stage: "Contact", success: hp >= 2, chance: 1.0 }},
        {{ stage: "Objective", success: score >= target*0.75, chance: 1.0 }},
        {{ stage: "Extraction", success: success, chance: 1.0 }},
      ];
      const result = {{
        version: 2,
        mode: "arena",
        mission_id: missionId,
        score,
        hp,
        success,
        deltas: {{
          intel_delta,
          supplies_delta,
          heat_delta,
          stability_delta,
          threat_delta,
        }},
        artifact_bonus: artifactBonus,
        stage_results,
        grade,
      }};
      const json = JSON.stringify(result);
      const enc = btoa(unescape(encodeURIComponent(json))).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'');
      window.__threadfall_result = enc;
    }}
    document.getElementById('finish').onclick = () => {{
      const enc = window.__threadfall_result;
      if (!enc) return;
      const url = new URL(window.location.href);
      url.searchParams.set("enc", enc);
      window.location.href = url.toString();
    }};
    window.addEventListener('keydown', (e)=> {{
      keys.add(e.key.toLowerCase());
      if (e.code === "Space") e.preventDefault();
    }});
    window.addEventListener('keyup', (e)=> keys.delete(e.key.toLowerCase()));

    let last = performance.now();
    function tick(now) {{
      const dt = Math.min(0.033, (now-last)/1000); last = now;
      if (!done) {{
        t -= dt;
        spawn -= dt;
        comboTimer -= dt;
        if (comboTimer <= 0) combo = Math.max(1.0, combo - dt*1.2);
        shake = Math.max(0, shake - dt*14);
        hitFlash = Math.max(0, hitFlash - dt*2.8);
        if (spawn <= 0) {{ spawn = 0.55 / difficultyMul; spawnEnemy(); }}
        const up = keys.has('w')||keys.has('arrowup');
        const dn = keys.has('s')||keys.has('arrowdown');
        const lf = keys.has('a')||keys.has('arrowleft');
        const rt = keys.has('d')||keys.has('arrowright');
        const sp = 240;
        player.vx = (rt-lf)*sp; player.vy = (dn-up)*sp;
        player.x = clamp(player.x + player.vx*dt, 10, canvas.width-10);
        player.y = clamp(player.y + player.vy*dt, 10, canvas.height-10);
        if (keys.has(' ') || keys.has('space')) {{
          if (!player._cd) player._cd = 0;
          player._cd -= dt;
          if (player._cd <= 0) {{ player._cd = 0.15; fire(); }}
        }}
        for (let i=bullets.length-1;i>=0;i--) {{
          const b=bullets[i]; b.x += b.vx*dt; b.y += b.vy*dt; b.life -= dt;
          if (b.life<=0||b.x>canvas.width+50) bullets.splice(i,1);
        }}
        for (let i=enemyBullets.length-1;i>=0;i--) {{
          const b=enemyBullets[i];
          b.x += b.vx*dt; b.y += b.vy*dt; b.life -= dt;
          if (b.life<=0 || b.x<-60 || b.x>canvas.width+60 || b.y<-60 || b.y>canvas.height+60) {{
            enemyBullets.splice(i,1);
            continue;
          }}
          const d = Math.hypot(b.x-player.x, b.y-player.y);
          if (d < 10) {{
            enemyBullets.splice(i,1);
            hp -= 1;
            hitFlash = 1.0;
            shake = 1.0;
            combo = 1.0;
            comboTimer = 0;
            beep(140, 0.08, "sawtooth", 0.06);
            if (hp <= 0) endEncounter();
          }}
        }}
        for (let i=enemies.length-1;i>=0;i--) {{
          const e=enemies[i];
          const dx=player.x-e.x, dy=player.y-e.y;
          const d=Math.hypot(dx,dy)||1;
          if (e.kind === "chaser") {{
            e.x += (dx/d)*e.sp*dt; e.y += (dy/d)*e.sp*dt;
          }} else if (e.kind === "flanker") {{
            // orbit-like movement: move toward but with perpendicular component
            const px = -dy/d, py = dx/d;
            e.angle += dt*0.9;
            const w = 0.55 + 0.25*Math.sin(e.angle);
            e.x += ((dx/d)*(1-w) + px*w) * e.sp*dt;
            e.y += ((dy/d)*(1-w) + py*w) * e.sp*dt;
          }} else {{
            // turret: slow drift + shooting
            e.x += (dx/d)*e.sp*dt*0.35; e.y += (dy/d)*e.sp*dt*0.35;
            e.cd -= dt;
            if (e.cd <= 0) {{
              e.cd = rand(0.65, 1.05)/difficultyMul;
              const bx = dx/d, by = dy/d;
              enemyBullets.push({{ x: e.x, y: e.y, vx: bx*220*difficultyMul, vy: by*220*difficultyMul, life: 2.2 }});
              beep(260, 0.05, "triangle", 0.03);
            }}
          }}
          if (d < e.r + 10) {{
            enemies.splice(i,1);
            hp -= 1;
            hitFlash = 1.0;
            shake = 1.0;
            combo = 1.0;
            comboTimer = 0;
            beep(110, 0.09, "sawtooth", 0.06);
            if (hp <= 0) endEncounter();
          }}
        }}
        for (let i=enemies.length-1;i>=0;i--) {{
          const e=enemies[i];
          for (let j=bullets.length-1;j>=0;j--) {{
            const b=bullets[j];
            const d=Math.hypot(e.x-b.x, e.y-b.y);
            if (d < e.r+4) {{
              enemies.splice(i,1);
              bullets.splice(j,1);
              const mult = 1.0 + Math.min(2.5, (combo-1.0));
              score += Math.round(5*mult);
              combo = Math.min(4.0, combo + 0.18);
              comboTimer = 1.1;
              lastKill = now;
              beep(780, 0.04, "square", 0.03);
              break;
            }}
          }}
        }}
        if (t <= 0) endEncounter();
      }}
      // Render
      const ox = (shake>0) ? (rand(-1,1)*shake*6) : 0;
      const oy = (shake>0) ? (rand(-1,1)*shake*6) : 0;
      ctx.setTransform(1,0,0,1,0,0);
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.translate(ox, oy);
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      for (let i=0;i<40;i++) ctx.fillRect((i*97)%canvas.width, (i*53)%canvas.height, 2, 2);
      // player
      ctx.beginPath(); ctx.arc(player.x, player.y, 10, 0, Math.PI*2);
      ctx.fillStyle = "#7ee7ff"; ctx.fill();
      // bullets
      ctx.fillStyle="#d4ff8a";
      bullets.forEach(b=> ctx.fillRect(b.x-3,b.y-1,7,3));
      // enemy bullets
      ctx.fillStyle="#ffcc6a";
      enemyBullets.forEach(b=> ctx.fillRect(b.x-2,b.y-2,4,4));
      // enemies
      enemies.forEach(e=> {{
        ctx.beginPath(); ctx.arc(e.x,e.y,e.r,0,Math.PI*2);
        ctx.fillStyle = e.kind==="turret" ? "#ffcc6a" : e.kind==="flanker" ? "#b38cff" : "#ff5b79";
        ctx.fill();
      }});
      if (hitFlash > 0) {{
        ctx.setTransform(1,0,0,1,0,0);
        ctx.fillStyle = "rgba(255,80,120," + (0.18*hitFlash).toFixed(3) + ")";
        ctx.fillRect(0,0,canvas.width,canvas.height);
      }}
      // hud
      document.getElementById('t').textContent = "TIME: " + Math.max(0,t).toFixed(1);
      document.getElementById('s').textContent = "SCORE: " + score;
      document.getElementById('hp').textContent = "HULL: " + hp;
      document.getElementById('combo').textContent = "COMBO: x" + combo.toFixed(1);
      const target = (diff==="Low"? 55: diff==="Elevated"? 75: diff==="Severe"? 95: 120);
      const grade = score >= target*1.35 ? "S" : score >= target*1.15 ? "A" : score >= target ? "B" : score >= target*0.75 ? "C" : "D";
      const gradeEl = document.getElementById('grade');
      gradeEl.textContent = "GRADE: " + grade;
      gradeEl.className = "pill " + (grade==="S"||grade==="A" ? "good" : grade==="D" ? "bad" : "");
      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  </script>
</body>
</html>
""".strip()


def ghosts_encounter_html(*, mission: dict[str, Any], context: dict[str, Any]) -> str:
    payload = {
        "version": 1,
        "mode": "chase",
        "mission_id": mission.get("mission_id", ""),
        "sector": mission.get("sector", ""),
        "difficulty": mission.get("difficulty", "Low"),
        "mission_type": mission.get("mission_type", "standard"),
        "sector_tags": list(mission.get("sector_tags", [])),
        "anomaly_tags": list(mission.get("anomaly_tags", [])),
        "world": {
            "threat_clock": int(context.get("threat_clock", 0)),
            "supplies": int(context.get("supplies", 0)),
            "sector_stability": int(context.get("sector_stability", 0)),
        },
        "sector_state": {
            "heat": int(context.get("heat", 0)),
            "stability": int(context.get("sector_local_stability", 0)),
        },
    }
    start = _b64url_encode(payload)
    title = f"Chase Encounter — {mission.get('sector', 'Unknown')}"
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; background:#070a15; color:#e6edf3; font-family: ui-sans-serif, system-ui; }}
    .wrap {{ padding: 12px 14px; }}
    .hud {{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; }}
    .pill {{ padding:6px 10px; border-radius:999px; background:#14203b; border:1px solid #2a3a74; font-weight:600; font-size:12px; }}
    canvas {{ display:block; margin-top:12px; width:100%; max-width:820px; height:520px; border-radius:14px; border:1px solid #2a3a74; background:#0b1020; }}
    .hint {{ opacity:0.85; font-size:12px; margin-top:8px; max-width:820px; }}
    .btn {{ margin-top:10px; padding:10px 12px; border-radius:12px; background:#2d4cff; border:none; color:white; font-weight:700; cursor:pointer; }}
    .btn:disabled {{ opacity:0.5; cursor:not-allowed; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hud">
      <div class="pill">MODE: CHASE</div>
      <div class="pill">MISSION: {mission.get("mission_type","standard")}</div>
      <div class="pill">DIFFICULTY: {mission.get("difficulty","Low")}</div>
      <div class="pill" id="t">TIME: 60.0</div>
      <div class="pill" id="intel">INTEL: 0/8</div>
      <div class="pill" id="hit">HITS: 0</div>
      <div class="pill" id="grade">GRADE: —</div>
    </div>
    <canvas id="c" width="820" height="520"></canvas>
    <div class="hint">Controls: WASD/Arrow keys. Collect intel nodes (cyan). Avoid hunters (red). Success = collect enough intel before time runs out.</div>
    <button class="btn" id="finish" disabled>Return result to Threadfall</button>
  </div>
  <script>
    const START = "{start}";
    const state = JSON.parse(atob(START.replace(/-/g,'+').replace(/_/g,'/').padEnd(Math.ceil(START.length/4)*4,'=')));
    const missionId = state.mission_id || "";
    const diff = state.difficulty || "Low";
    const heat = (state.sector_state && typeof state.sector_state.heat === "number") ? state.sector_state.heat : 0;
    const threatClock = (state.world && typeof state.world.threat_clock === "number") ? state.world.threat_clock : 0;
    const missionType = state.mission_type || "standard";
    const need = diff==="Low"? 6 : diff==="Elevated"? 7 : diff==="Severe"? 8 : 9;
    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    const keys = new Set();
    // tiny audio helper (no assets)
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = AudioCtx ? new AudioCtx() : null;
    function beep(freq, dur, type="sine", gain=0.05) {{
      if (!audioCtx) return;
      const t0 = audioCtx.currentTime;
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = type;
      o.frequency.setValueAtTime(freq, t0);
      g.gain.setValueAtTime(gain, t0);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(t0); o.stop(t0 + dur);
    }}
    const grid = {{ w: 20, h: 13, size: 40 }};
    const walls = new Set();
    function key(x,y){{ return x+','+y; }}
    // simple static maze
    for (let x=0;x<grid.w;x++) {{ walls.add(key(x,0)); walls.add(key(x,grid.h-1)); }}
    for (let y=0;y<grid.h;y++) {{ walls.add(key(0,y)); walls.add(key(grid.w-1,y)); }}
    for (let x=2;x<18;x++) {{ if (x!==10) walls.add(key(x,4)); }}
    for (let x=2;x<18;x++) {{ if (x!==6) walls.add(key(x,8)); }}
    for (let y=2;y<11;y++) {{ if (y!==6) walls.add(key(6,y)); if (y!==7) walls.add(key(14,y)); }}

    function isWall(x,y){{ return walls.has(key(x,y)); }}
    function clamp(v,a,b){{ return Math.max(a, Math.min(b,v)); }}
    const player = {{ x: 2, y: 2 }};
    // 3 hunter archetypes:
    // - intercept: aims ahead of player direction
    // - tracker: goes to last seen player position (updates on pings)
    // - patrol: loops waypoints and switches to chase on ping
    const hunters = [
      {{x:17,y:10, kind:"intercept"}},
      {{x:17,y:2, kind:"tracker", lx:17, ly:2}},
      {{x:10,y:10, kind:"patrol", wp:0}},
    ];
    let intel = 0;
    let hits = 0;
    let t = 60.0;
    t -= Math.min(12, Math.floor(heat/10));
    t -= (threatClock >= 70 ? 6 : threatClock >= 40 ? 3 : 0);
    let done = false;
    let flash = 0;
    let ping = 0;
    let lastDx = 0, lastDy = 0;
    const intelNodes = [];
    function placeIntel() {{
      intelNodes.length=0;
      const spots = [];
      for (let y=1;y<grid.h-1;y++) for (let x=1;x<grid.w-1;x++) if (!isWall(x,y)) spots.push({{x,y}});
      for (let i=0;i<12;i++) {{
        const s = spots[Math.floor(Math.random()*spots.length)];
        intelNodes.push({{x:s.x,y:s.y,alive:true}});
      }}
    }}
    placeIntel();

    function moveEnt(ent, dx, dy) {{
      const nx = ent.x + dx, ny = ent.y + dy;
      if (!isWall(nx, ny)) {{ ent.x = nx; ent.y = ny; }}
    }}
    function stepToward(h, tx, ty) {{
      const options = [
        {{dx:1,dy:0}}, {{dx:-1,dy:0}}, {{dx:0,dy:1}}, {{dx:0,dy:-1}}
      ].filter(o=> !isWall(h.x+o.dx, h.y+o.dy));
      options.sort((a,b)=> {{
        const da = Math.abs((h.x+a.dx)-tx) + Math.abs((h.y+a.dy)-ty);
        const db = Math.abs((h.x+b.dx)-tx) + Math.abs((h.y+b.dy)-ty);
        return da-db;
      }});
      const pick = options[0] || {{dx:0,dy:0}};
      h.x += pick.dx; h.y += pick.dy;
    }}
    const waypoints = [{{x:2,y:10}}, {{x:10,y:10}}, {{x:17,y:10}}, {{x:17,y:2}}, {{x:10,y:2}}, {{x:2,y:2}}];
    function hunterStep(h) {{
      if (h.kind === "patrol") {{
        const w = waypoints[h.wp % waypoints.length];
        stepToward(h, w.x, w.y);
        if (h.x===w.x && h.y===w.y) h.wp = (h.wp+1)%waypoints.length;
        if (ping > 0) stepToward(h, player.x, player.y);
        return;
      }}
      if (h.kind === "tracker") {{
        // update last known on ping or close proximity
        if (ping > 0 || (Math.abs(h.x-player.x)+Math.abs(h.y-player.y) <= 6)) {{
          h.lx = player.x; h.ly = player.y;
        }}
        stepToward(h, h.lx, h.ly);
        return;
      }}
      // intercept
      const tx = clamp(player.x + lastDx*3, 1, grid.w-2);
      const ty = clamp(player.y + lastDy*3, 1, grid.h-2);
      stepToward(h, tx, ty);
    }}

    function endEncounter() {{
      done = true;
      document.getElementById('finish').disabled = false;
      const success = intel >= need && hits < 3;
      const bonusIntel = Math.max(0, Math.round(intel - hits));
      // Mission-type shaping: extraction rewards cleaner extractions; survey/containment rarely yield artifacts.
      let artifactBonus = 0;
      if (missionType === "extraction") {{
        artifactBonus = (intel >= need+1 && hits <= 1) ? 1 : 0;
      }} else if (missionType === "standard" || missionType === "deep_dive") {{
        artifactBonus = (intel >= need+2 && hits===0) ? 1 : 0;
      }}
      const grade = success ? (hits===0 && intel>=need+2 ? "S" : hits<=1 ? "A" : "B") : (intel >= need-1 ? "C" : "D");
      const intel_delta = bonusIntel;
      const supplies_delta = 0;
      const heat_delta = Math.max(0, Math.min(10, Math.floor((need + intel)/3) + hits));
      const stability_delta = (success ? Math.min(5, 1 + Math.floor(intel/3)) : -Math.min(6, 2 + hits));
      const threat_delta = (success ? -Math.min(5, 1 + Math.floor(intel/4)) : Math.min(8, 3 + hits*2));
      const stage_results = [
        {{ stage: "Approach", success: hits === 0, chance: 1.0 }},
        {{ stage: "Contact", success: hits <= 1, chance: 1.0 }},
        {{ stage: "Objective", success: intel >= need-1, chance: 1.0 }},
        {{ stage: "Extraction", success: success, chance: 1.0 }},
      ];
      const result = {{
        version: 2,
        mode: "chase",
        mission_id: missionId,
        intel,
        hits,
        success,
        deltas: {{
          intel_delta,
          supplies_delta,
          heat_delta,
          stability_delta,
          threat_delta,
        }},
        artifact_bonus: artifactBonus,
        stage_results,
        grade,
      }};
      const json = JSON.stringify(result);
      const enc = btoa(unescape(encodeURIComponent(json))).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'');
      window.__threadfall_result = enc;
    }}
    document.getElementById('finish').onclick = () => {{
      const enc = window.__threadfall_result;
      if (!enc) return;
      const url = new URL(window.location.href);
      url.searchParams.set("enc", enc);
      window.location.href = url.toString();
    }};

    window.addEventListener('keydown', (e)=> keys.add(e.key.toLowerCase()));
    window.addEventListener('keyup', (e)=> keys.delete(e.key.toLowerCase()));

    let last = performance.now();
    let moveCd = 0;
    let hunterCd = 0;
    function tick(now) {{
      const dt = Math.min(0.05, (now-last)/1000); last = now;
      if (!done) {{
        t -= dt;
        moveCd -= dt;
        hunterCd -= dt;
        flash = Math.max(0, flash - dt*2.8);
        ping = Math.max(0, ping - dt*1.2);
        if (moveCd <= 0) {{
          moveCd = 0.09;
          const up = keys.has('w')||keys.has('arrowup');
          const dn = keys.has('s')||keys.has('arrowdown');
          const lf = keys.has('a')||keys.has('arrowleft');
          const rt = keys.has('d')||keys.has('arrowright');
          lastDx = 0; lastDy = 0;
          if (up) {{ moveEnt(player,0,-1); lastDy=-1; }}
          else if (dn) {{ moveEnt(player,0,1); lastDy=1; }}
          else if (lf) {{ moveEnt(player,-1,0); lastDx=-1; }}
          else if (rt) {{ moveEnt(player,1,0); lastDx=1; }}
          intelNodes.forEach(n=> {{
            if (n.alive && n.x===player.x && n.y===player.y) {{ n.alive=false; intel+=1; }}
          }});
          // periodic ping reveals you (acoustic-style tension)
          if (Math.floor(t*2) % 12 === 0) {{
            ping = 1.0;
          }}
        }}
        if (hunterCd <= 0) {{
          hunterCd = 0.22;
          hunters.forEach(hunterStep);
          hunters.forEach(h=> {{
            if (h.x===player.x && h.y===player.y) {{
              hits+=1; player.x=2; player.y=2;
              flash = 1.0;
              ping = 1.0;
              beep(140, 0.09, "sawtooth", 0.06);
            }}
          }});
        }}
        if (t <= 0 || hits >= 3) endEncounter();
        if (intel >= need+3) endEncounter();
      }}
      // render
      ctx.clearRect(0,0,canvas.width,canvas.height);
      for (let y=0;y<grid.h;y++) for (let x=0;x<grid.w;x++) {{
        if (isWall(x,y)) {{
          ctx.fillStyle = "#1c2a57";
          ctx.fillRect(x*grid.size, y*grid.size, grid.size, grid.size);
        }}
      }}
      intelNodes.forEach(n=> {{
        if (!n.alive) return;
        ctx.fillStyle = "#7ee7ff";
        ctx.beginPath();
        ctx.arc(n.x*grid.size+grid.size/2, n.y*grid.size+grid.size/2, 7, 0, Math.PI*2);
        ctx.fill();
      }});
      // player
      ctx.fillStyle="#d4ff8a";
      ctx.fillRect(player.x*grid.size+10, player.y*grid.size+10, grid.size-20, grid.size-20);
      // hunters
      hunters.forEach(h=> {{
        ctx.fillStyle = h.kind==="intercept" ? "#ff5b79" : h.kind==="tracker" ? "#b38cff" : "#ffcc6a";
        ctx.beginPath();
        ctx.arc(h.x*grid.size+grid.size/2, h.y*grid.size+grid.size/2, 12, 0, Math.PI*2);
        ctx.fill();
      }});
      if (ping > 0) {{
        ctx.fillStyle = "rgba(126,231,255," + (0.08*ping).toFixed(3) + ")";
        ctx.fillRect(0,0,canvas.width,canvas.height);
      }}
      if (flash > 0) {{
        ctx.fillStyle = "rgba(255,80,120," + (0.18*flash).toFixed(3) + ")";
        ctx.fillRect(0,0,canvas.width,canvas.height);
      }}
      document.getElementById('t').textContent = "TIME: " + Math.max(0,t).toFixed(1);
      document.getElementById('intel').textContent = "INTEL: " + intel + "/" + need;
      document.getElementById('hit').textContent = "HITS: " + hits;
      const grade = (intel>=need && hits<3) ? (hits===0 && intel>=need+2 ? "S" : hits<=1 ? "A" : "B") : (intel >= need-1 ? "C" : "D");
      document.getElementById('grade').textContent = "GRADE: " + grade;
      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  </script>
</body>
</html>
""".strip()

