import time

import streamlit as st
import streamlit.components.v1 as components

from threadfall.game import (
    append_log,
    autoplay_state,
    autoplay_player_turn,
    bootstrap_state,
    build_mission,
    configure_ai_agents,
    configure_autoplay,
    current_device_label,
    ensure_autonomous_stack_defaults,
    evaluate_endgame,
    game_status,
    init_rival_learning,
    init_services,
    remember_current_mission,
    reset_campaign,
    resolve_current_mission,
    run_autonomous_stack_tick,
    run_ml_swarm_until_endgame,
    stop_autoplay,
)
from threadfall.ui import (
    render_activity_feed,
    render_agent_summary,
    render_ai_expedition_feed,
    render_autoplay_status,
    render_command_bar,
    render_counter_missions,
    render_decision_engine_status,
    render_endgame_summary,
    render_events,
    render_faction_state,
    render_history,
    render_inventory,
    render_mas_panel,
    render_memory_results,
    render_negotiation_feed,
    render_rival_units,
    render_sector_control,
    render_status_cards,
    render_topology_map,
    render_timeline_charts,
    render_world_state,
)
from threadfall.encounters import decode_encounter_query, ghosts_encounter_html, robotron_encounter_html


st.set_page_config(
    page_title="Threadfall Prototype",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def page_overview() -> None:
    st.title("Threadfall")
    st.caption("Agentic anomaly expedition prototype built with Python, Streamlit, and Qdrant.")

    bootstrap_state()
    services = init_services()
    render_command_bar(services, ended=st.session_state["endgame"]["ended"])

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Mission Fantasy")
        st.write(
            """
            Descend into a living ruin, document reality-bending anomalies, survive adaptive threats,
            and bring back knowledge that changes what future expeditions can discover.
            """
        )

        st.subheader("Current Build")
        st.write(
            """
            This prototype now includes the first progression loop: mission generation,
            expedition resolution, faction pressure, artifact inventory, and vector retrieval
            for world knowledge.
            """
        )

        st.subheader("Squad Setup")
        st.text_input(
            "Squad name",
            help="This tag is attached to expedition memories.",
            key="squad_name",
        )
        st.text_input(
            "Operator name",
            key="operator_name",
        )

        st.subheader("AI Rival Units")
        current_cfg = int(st.session_state.get("ai_config", {}).get("num_agents", 0))
        current_widget = st.session_state.get("ai_num_agents_slider", current_cfg)
        num_agents = st.slider(
            "Number of rival AI units",
            min_value=0,
            max_value=6,
            value=int(current_widget),
            step=1,
            help="Rival units run their own expeditions during autonomous simulation ticks and can win the campaign.",
            key="ai_num_agents_slider",
        )
        st.caption("Changing rival count requires restarting the campaign (your current run will be reset).")
        if st.button("Apply & Restart Campaign", use_container_width=True, key="ai_apply_restart"):
            st.session_state["ai_config"] = {"num_agents": int(num_agents)}
            configure_ai_agents(int(num_agents))
            reset_campaign()
            st.rerun()

        agents = st.session_state.get("ai_agents", [])
        if agents:
            st.markdown("#### Rival Standings")
            for a in agents:
                st.markdown(
                    f"- **{a['name']}** · intel `{a.get('intel', 0)}` · artifacts `{a.get('artifacts', 0)}` · last `{a.get('last_outcome', 'None')}`"
                )

    with right:
        render_status_cards(services)
        status = game_status()
        st.subheader("Campaign Status")
        st.metric("State", status["state"])
        st.caption(status["detail"])

    st.subheader("Ruin Topology Map")
    render_topology_map(
        sector_control=st.session_state["sector_control"],
        factions=st.session_state["factions"],
        map_slot="overview",
    )

    st.subheader("Expedition Log")
    for entry in st.session_state["log"][-8:][::-1]:
        st.markdown(f"- {entry}")


def page_mission() -> None:
    st.title("Expedition Console")
    bootstrap_state()
    services = init_services()
    evaluate_endgame()
    ended = st.session_state["endgame"]["ended"]
    render_command_bar(services, ended=ended)

    # Receive encounter result via query param and store it.
    if "enc" in st.query_params:
        try:
            st.session_state["pending_encounter"] = decode_encounter_query(st.query_params["enc"])
        except Exception:
            st.session_state["pending_encounter"] = {"error": "invalid encounter payload"}
        st.query_params.clear()

    controls, mission_col = st.columns([0.9, 1.4])

    with controls:
        st.subheader("Mission Controls")
        mission = st.session_state["current_mission"]
        world = st.session_state["world_state"]
        mission_type = mission.get("mission_type", "standard")
        max_commit = 0
        if mission_type in {"extraction", "containment", "standard", "counter"}:
            max_commit = 2
        elif mission_type == "survey":
            max_commit = 1
        max_commit = min(max_commit, int(world.get("supplies", 0)))

        if max_commit == 0:
            st.session_state["mission_commit_supplies"] = 0
            st.caption("No extra supplies available to commit this cycle.")
        else:
            st.session_state["mission_commit_supplies"] = st.slider(
                "Commit supplies (boost success odds)",
                min_value=0,
                max_value=max_commit,
                value=min(int(st.session_state.get("mission_commit_supplies", 0)), max_commit),
                step=1,
                help="Spend extra supplies this cycle to improve expedition odds. This is consumed whether you succeed or fail.",
                key="mission_commit_supplies_slider",
                disabled=ended,
            )
            st.caption(f"Available supplies: {world.get('supplies', 0)} · committing: {st.session_state['mission_commit_supplies']}")

        if st.button("Generate New Expedition", use_container_width=True, disabled=ended, key="mission_generate"):
            st.session_state["current_mission"] = build_mission(
                squad_name=st.session_state["squad_name"]
            )
            append_log("Generated a fresh expedition profile.")

        if st.button("Store Mission Memory", use_container_width=True, disabled=ended, key="mission_store"):
            mission = st.session_state["current_mission"]
            memory_id = remember_current_mission(services.memory, mission)
            append_log(f"Stored expedition memory: {memory_id}")

        st.markdown("### Player Autopilot")
        st.caption("Let an agent run your expedition loop automatically (commit supplies + resolve).")
        enabled_autopilot = st.checkbox(
            "Enable player autopilot",
            value=bool(st.session_state.get("player_autopilot_enabled", False)),
            key="player_autopilot_enabled",
            disabled=ended,
        )
        if enabled_autopilot and not ended:
            steps = st.number_input(
                "Autopilot turns",
                min_value=1,
                max_value=50,
                value=int(st.session_state.get("player_autopilot_steps", 5)),
                step=1,
                key="player_autopilot_steps",
                help="Runs multiple player expeditions in a row (each advances cycle).",
            )
            if st.button("Run 1 Autopilot Turn", type="primary", use_container_width=True, key="auto_run_one"):
                result = autoplay_player_turn(services.memory, services.decision_engine)
                st.success(f"Autopilot resolved: cycle {result['cycle']} · {result['outcome']}")
                st.rerun()
            if st.button(f"Run Autopilot ({int(steps)} turns)", use_container_width=True, key="auto_run_many"):
                last = None
                for _ in range(int(steps)):
                    if st.session_state["endgame"]["ended"]:
                        break
                    last = autoplay_player_turn(services.memory, services.decision_engine)
                if last:
                    st.success(f"Autopilot finished on cycle {last['cycle']} · {last['outcome']}")
                st.rerun()

        st.markdown("### Encounters (Action Layer)")
        st.caption("Play a short encounter to determine mission success/failure, then apply it to resolve the expedition.")
        if st.button("Play Arena Encounter (A)", use_container_width=True, disabled=ended, key="enc_a"):
            st.session_state["encounter_mode"] = "arena"
        if st.button("Play Chase Encounter (B)", use_container_width=True, disabled=ended, key="enc_b"):
            st.session_state["encounter_mode"] = "chase"

        mode = st.session_state.get("encounter_mode")
        if mode in {"arena", "chase"}:
            sector_state = st.session_state.get("sector_control", {}).get(mission.get("sector", ""), {})
            ctx_payload = {
                "threat_clock": int(world.get("threat_clock", 0)),
                "supplies": int(world.get("supplies", 0)),
                "sector_stability": int(world.get("sector_stability", 0)),
                "heat": int(sector_state.get("heat", 0)),
                "sector_local_stability": int(sector_state.get("stability", 0)),
            }
            html = (
                robotron_encounter_html(mission=mission, context=ctx_payload)
                if mode == "arena"
                else ghosts_encounter_html(mission=mission, context=ctx_payload)
            )
            components.html(html, height=680, scrolling=False)

        pending = st.session_state.get("pending_encounter")
        if isinstance(pending, dict) and pending.get("mission_id") == mission.get("mission_id"):
            st.success(
                f"Encounter captured: mode `{pending.get('mode')}` · success `{pending.get('success')}` · "
                f"deltas `{pending.get('deltas', {})}`"
            )
            if st.button("Apply Encounter Result (Resolve Expedition)", type="primary", use_container_width=True, key="enc_apply"):
                result = resolve_current_mission(services.memory)
                st.success(f"Resolved via encounter: {result['outcome']}")
                st.session_state["encounter_mode"] = None
                st.rerun()

        if st.button(
            "Resolve Expedition",
            type="primary",
            use_container_width=True,
            disabled=ended,
            key="mission_resolve",
        ):
            result = resolve_current_mission(services.memory)
            st.success(f"Cycle {result['cycle']} resolved: {result['outcome']}")

        if ended:
            st.warning("Campaign complete. Open Campaign Summary to review the final result or reset from Operations.")

        st.markdown("### Runtime")
        st.write(f"Embedding device: `{current_device_label(services.embedder)}`")
        st.write(f"Memory backend: `{services.memory.backend_name}`")

    with mission_col:
        mission = st.session_state["current_mission"]
        st.subheader(mission["title"])
        st.caption(f"Mission type: `{mission.get('mission_type', 'standard')}`")
        if mission.get("mission_label"):
            st.caption(f"Profile: **{mission['mission_label']}**")
        tags_line = " · ".join(
            [f"sector: {', '.join(mission.get('sector_tags', []))}" if mission.get("sector_tags") else ""]
            + [f"anomaly: {', '.join(mission.get('anomaly_tags', []))}" if mission.get("anomaly_tags") else ""]
        ).strip(" ·")
        if tags_line:
            st.caption(tags_line)
        top = st.columns(4)
        top[0].metric("Sector", mission["sector"])
        top[1].metric("Threat", mission["threat"])
        top[2].metric("Objective", mission["objective"])
        top[3].metric("Difficulty", mission["difficulty"])

        st.caption(f"Sponsoring faction: {mission['sponsor']}")
        st.caption(f"Rival faction: {mission['rival']}")

        st.markdown("### Anomaly Profile")
        st.write(mission["anomaly"])

        st.markdown("### Faction Directive")
        st.write(mission["directive"])
        for tag in mission["pressure_tags"]:
            st.markdown(f"- {tag}")

        st.markdown("### Artifact Signal")
        st.write(
            f"Potential recovery: **{mission['artifact']['name']}** ({mission['artifact']['rarity']})"
        )
        st.caption(mission["artifact"]["effect"])

        st.markdown("### Field Brief")
        st.write(mission["brief"])

        st.markdown("### Suggested Squad Response")
        for line in mission["response"]:
            st.markdown(f"- {line}")

        st.markdown("### Mission Stages")
        stages = mission.get("stages", [])
        if stages:
            for stage in stages:
                st.markdown(f"- **{stage.get('name', 'Stage')}** — {stage.get('focus', '')}")
        else:
            st.caption("No stage data available.")

        last = st.session_state.get("last_resolution")
        if last:
            st.markdown("### Last Resolution")
            st.markdown(f"**Cycle {last['cycle']}** · `{last['outcome']}` · `{last.get('mission_type', 'standard')}`")
            stage_results = last.get("stage_results", [])
            if stage_results:
                for item in stage_results:
                    marker = "✅" if item.get("success") else "❌"
                    st.markdown(f"- {marker} **{item.get('stage', 'Stage')}** (chance `{item.get('chance', 0):.3f}`)")
            st.caption(last.get("summary", ""))


def page_operations() -> None:
    st.title("Operations")
    bootstrap_state()
    evaluate_endgame()
    services = init_services()
    render_command_bar(services, ended=st.session_state["endgame"]["ended"])

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("World State")
        render_world_state(st.session_state["world_state"])

        st.subheader("Sector Control")
        render_sector_control(st.session_state["sector_control"])

        st.subheader("Expedition History")
        render_history(st.session_state["history"])

    with right:
        st.subheader("Faction Pressure")
        render_faction_state(st.session_state["factions"])

        st.subheader("Recovered Artifacts")
        render_inventory(st.session_state["inventory"])

        st.subheader("Queued Counter-Missions")
        render_counter_missions(st.session_state["counter_missions"])

        render_mas_panel()

        if st.button("Reset Campaign", use_container_width=True, key="ops_reset_campaign"):
            reset_campaign()
            st.warning("Campaign save reset.")


def page_memory() -> None:
    st.title("Memory Archive")
    bootstrap_state()
    services = init_services()
    evaluate_endgame()
    render_command_bar(services, ended=st.session_state["endgame"]["ended"])

    query = st.text_input(
        "Search expedition memories",
        value=st.session_state.get("last_query", "gravity fracture archive chamber"),
        key="memory_query",
    )
    st.session_state["last_query"] = query

    if st.button("Query Memory Archive", use_container_width=True, key="memory_query_btn"):
        st.session_state["memory_results"] = services.memory.search(query, limit=5)
        append_log(f"Queried memory archive for: {query}")

    render_memory_results(st.session_state["memory_results"])


def page_dashboard() -> None:
    st.title("Agent Activity Dashboard")
    bootstrap_state()
    services = init_services()
    evaluate_endgame()
    ended = st.session_state["endgame"]["ended"]
    render_command_bar(services, ended=ended)

    controls, summary = st.columns([0.9, 1.4])

    with controls:
        sim_toast = st.session_state.pop("_dashboard_sim_toast", None)
        if sim_toast:
            st.success(sim_toast)
        st.subheader("Simulation Controls")
        with st.expander("Autonomous ML stack (defaults on)", expanded=False):
            s = ensure_autonomous_stack_defaults()
            s["enabled"] = st.checkbox(
                "Run AI player + factions on each autoplay tick",
                value=bool(s.get("enabled", True)),
                key="dash_auto_stack_enabled",
            )
            s["use_ollama_player_commit"] = st.checkbox(
                "Ollama advises player supply commits",
                value=bool(s.get("use_ollama_player_commit", True)),
                key="dash_auto_stack_llm",
            )
            s["player_expeditions_per_tick"] = int(
                st.number_input(
                    "Player expeditions per tick",
                    min_value=0,
                    max_value=3,
                    value=int(s.get("player_expeditions_per_tick", 1)),
                    step=1,
                    key="dash_auto_stack_pe",
                )
            )
            s["faction_rounds_per_tick"] = int(
                st.number_input(
                    "Faction agent cycles per tick",
                    min_value=1,
                    max_value=8,
                    value=int(s.get("faction_rounds_per_tick", 1)),
                    step=1,
                    key="dash_auto_stack_fr",
                )
            )
            st.session_state["autonomous_stack"] = s
            st.caption(
                "Rival units default to **bandit** learning; factions use **MAS** (LLM / heuristic / bandit). "
                "Autoplay ticks run the full stack while the campaign is Active."
            )
        with st.expander("ML agent swarm (end-to-end)", expanded=False):
            st.caption(
                "Runs stack ticks in one shot until **endgame** or **max ticks**: player autoplay (no encounter UI) "
                "+ dense faction rounds + rival competitors each cycle. Restores your stack presets afterward."
            )
            swarm_cap = int(
                st.number_input(
                    "Max stack ticks",
                    min_value=10,
                    max_value=8000,
                    value=2000,
                    step=50,
                    key="dash_swarm_cap",
                    help="Safety cap: each tick can run up to 3 player missions and 8 faction inner rounds.",
                )
            )
            swarm_throughput = st.checkbox(
                "Max throughput (3 player expeditions / tick, 8 faction rounds / tick)",
                value=True,
                key="dash_swarm_throughput",
            )
            swarm_bandit = st.checkbox(
                "All factions → MAS bandit for this run (no per-faction Ollama; fastest)",
                value=False,
                key="dash_swarm_bandit",
            )
            swarm_restore = st.checkbox(
                "Restore autonomous stack + MAS after run",
                value=True,
                key="dash_swarm_restore",
            )
            if st.button(
                "Run ML swarm to endgame",
                type="primary",
                use_container_width=True,
                disabled=ended,
                key="dash_swarm_run",
            ):
                with st.spinner("Running autonomous ML swarm…"):
                    out = run_ml_swarm_until_endgame(
                        services.memory,
                        services.decision_engine,
                        max_stack_ticks=swarm_cap,
                        max_throughput=swarm_throughput,
                        mas_bandit_only=swarm_bandit,
                        restore_stack_after=swarm_restore,
                        restore_mas_after=swarm_restore,
                    )
                _swarm_tail = (
                    "Hit tick cap while Active."
                    if out["hit_cap"]
                    else f"Ended: {out.get('result') or 'Active'}."
                )
                st.session_state["_dashboard_sim_toast"] = (
                    f"Swarm: {out['ticks']} tick(s), {out['player_resolutions']} player resolution(s), "
                    f"{out['faction_rows']} faction row(s). {_swarm_tail}"
                )
                st.rerun()

        if st.button(
            "Run 1 Agent Cycle",
            type="primary",
            use_container_width=True,
            disabled=ended,
            key="dash_run_1",
        ):
            tick = run_autonomous_stack_tick(services.memory, services.decision_engine)
            n_p, n_f = len(tick.get("player", [])), len(tick.get("factions", []))
            append_log(f"Manual stack tick: {n_p} player expedition(s), {n_f} faction decision(s).")
            st.session_state["_dashboard_sim_toast"] = (
                f"Executed 1 stack tick ({n_p} player resolutions · {n_f} faction rows)."
            )
            st.rerun()

        if st.button("Run 5 Agent Cycles", use_container_width=True, disabled=ended, key="dash_run_5"):
            total_p = total_f = 0
            for _ in range(5):
                tick = run_autonomous_stack_tick(services.memory, services.decision_engine)
                total_p += len(tick.get("player", []))
                total_f += len(tick.get("factions", []))
                if st.session_state["endgame"]["ended"]:
                    break
            append_log(f"Manual stack x5: {total_p} player expedition(s), {total_f} faction decision(s).")
            st.session_state["_dashboard_sim_toast"] = (
                f"Executed 5 stack ticks ({total_p} player resolutions · {total_f} faction rows)."
            )
            st.rerun()

        st.markdown("### Autoplay")
        ticks = st.number_input("Ticks", min_value=1, max_value=50, value=5, step=1, key="dash_ticks")
        rounds_per_tick = st.number_input(
            "Rounds per tick", min_value=1, max_value=5, value=1, step=1, key="dash_rounds_per_tick"
        )
        delay_seconds = st.slider(
            "Delay (seconds)",
            min_value=0.2,
            max_value=3.0,
            value=0.8,
            step=0.1,
            key="dash_delay_seconds",
        )
        start_col, stop_col = st.columns(2)
        if start_col.button("Start Autoplay", use_container_width=True, disabled=ended, key="dash_autoplay_start"):
            configure_autoplay(True, int(ticks), int(rounds_per_tick), float(delay_seconds))
            st.success("Autoplay armed.")
        if stop_col.button("Stop", use_container_width=True, key="dash_autoplay_stop"):
            stop_autoplay()
            st.warning("Autoplay stopped.")

        status = game_status()
        st.markdown("### Status")
        st.metric("Campaign State", status["state"])
        st.caption(status["detail"])
        st.markdown("### Decision Engine")
        render_decision_engine_status(services.decision_engine)
        render_autoplay_status(autoplay_state())
        if ended:
            st.warning("Campaign complete. Autonomous simulation is locked.")

    with summary:
        st.subheader("Faction Overview")
        render_agent_summary(st.session_state["factions"])
        st.subheader("Rival Units")
        render_rival_units(st.session_state.get("ai_agents", []))
        st.subheader("Live Events")
        render_events(st.session_state["events"])

    st.subheader("Current World State")
    render_world_state(st.session_state["world_state"])

    st.subheader("Ruin Topology Map")
    render_topology_map(
        sector_control=st.session_state["sector_control"],
        factions=st.session_state["factions"],
        map_slot="dashboard",
    )

    st.subheader("Timeline Charts")
    render_timeline_charts(st.session_state["snapshots"])

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Live Activity Feed")
        render_activity_feed(st.session_state["activities"])
        st.subheader("Rival Expedition Feed")
        render_ai_expedition_feed(st.session_state["history"])
    with right:
        st.subheader("Faction State")
        render_faction_state(st.session_state["factions"])

    st.subheader("Negotiation Feed")
    render_negotiation_feed(st.session_state["negotiations"])

    autoplay = autoplay_state()
    if autoplay["enabled"] and autoplay["remaining_ticks"] > 0 and game_status()["state"] == "Active":
        time.sleep(autoplay["delay_seconds"])
        n_p = n_f = 0
        for _ in range(max(1, int(autoplay["rounds_per_tick"]))):
            if st.session_state["endgame"]["ended"]:
                break
            tick = run_autonomous_stack_tick(services.memory, services.decision_engine)
            n_p += len(tick.get("player", []))
            n_f += len(tick.get("factions", []))
        st.session_state["autoplay_remaining_ticks"] = max(0, autoplay["remaining_ticks"] - 1)
        append_log(f"Autoplay tick: {n_p} player expedition(s), {n_f} faction decision(s).")
        if st.session_state["autoplay_remaining_ticks"] == 0:
            st.session_state["autoplay_enabled"] = False
        st.rerun()


def page_summary() -> None:
    st.title("Campaign Summary")
    bootstrap_state()
    evaluate_endgame()
    endgame = st.session_state["endgame"]
    services = init_services()
    render_command_bar(services, ended=endgame["ended"])

    if not endgame["ended"]:
        st.info("The campaign is still active. End conditions have not been met yet.")
        return

    render_endgame_summary(
        endgame=endgame,
        history=st.session_state["history"],
        factions=st.session_state["factions"],
        sector_control=st.session_state["sector_control"],
        inventory=st.session_state["inventory"],
    )
    st.subheader("Campaign Timeline")
    render_timeline_charts(st.session_state["snapshots"])


def page_training() -> None:
    st.title("Training Lab")
    bootstrap_state()
    services = init_services()
    init_rival_learning()
    ended = st.session_state["endgame"]["ended"]
    render_command_bar(services, ended=ended)

    st.subheader("Rival Learning Policy")
    policy = st.session_state["rival_policy"]
    mode = st.selectbox(
        "Policy mode",
        options=["heuristic", "bandit", "qlearn"],
        index=["heuristic", "bandit", "qlearn"].index(policy.get("mode", "heuristic")),
        key="train_policy_mode",
    )
    policy["mode"] = mode
    policy["epsilon"] = st.slider("Exploration (epsilon)", 0.0, 0.5, float(policy.get("epsilon", 0.09)), 0.01, key="train_eps")
    policy["alpha"] = st.slider("Q-learning alpha", 0.01, 0.6, float(policy.get("alpha", 0.07)), 0.01, key="train_alpha")
    policy["gamma"] = st.slider("Q-learning gamma", 0.5, 0.99, float(policy.get("gamma", 0.86)), 0.01, key="train_gamma")
    policy["competitor_reward_cap"] = st.slider(
        "Competitor reward cap (per expedition tick)",
        6.0,
        40.0,
        float(policy.get("competitor_reward_cap", 22.0)),
        1.0,
        help="Clamps shaped reward before bandit/Q updates—slows runaway learning.",
        key="train_reward_cap",
    )
    policy["q_value_clip"] = st.slider(
        "Q / bandit value clip",
        8.0,
        80.0,
        float(policy.get("q_value_clip", 35.0)),
        1.0,
        help="Absolute cap on stored action values after each update.",
        key="train_q_clip",
    )
    st.session_state["rival_policy"] = policy

    st.subheader("Run Training Steps (advances rivals only)")
    steps = st.number_input("Steps", min_value=1, max_value=500, value=25, step=1, key="train_steps")
    if st.button("Run Training", type="primary", use_container_width=True, key="train_run"):
        from threadfall.game import simulate_ai_competitors

        simulate_ai_competitors(services.memory, per_tick=int(steps))
        st.success(f"Ran {steps} rival training steps.")
        st.rerun()

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Rival Standings")
        render_rival_units(st.session_state.get("ai_agents", []))
    with cols[1]:
        st.subheader("Recent Rival Expeditions")
        render_ai_expedition_feed(st.session_state["history"])

    st.subheader("Learned Tables (debug)")
    if mode == "bandit":
        st.json(st.session_state.get("rival_bandit", {}))
    elif mode == "qlearn":
        st.json(st.session_state.get("rival_q", {}))
    else:
        st.caption("Heuristic mode has no learned table.")


bootstrap_state()

pages = {
    "Mission Control": [
        st.Page(page_overview, title="Overview", icon="🧭", default=True),
        st.Page(page_mission, title="Expedition", icon="🧭"),
        st.Page(page_operations, title="Operations", icon="📡"),
        st.Page(page_dashboard, title="Agent Dashboard", icon="🤖"),
        st.Page(page_memory, title="Memory Archive", icon="🧠"),
        st.Page(page_training, title="Training Lab", icon="🧪"),
        st.Page(page_summary, title="Campaign Summary", icon="🏁"),
    ]
}

router = st.navigation(pages)
router.run()
