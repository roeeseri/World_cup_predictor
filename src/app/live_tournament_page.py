from __future__ import annotations

import pandas as pd
import streamlit as st

from src.tournament.live_simulator import (
    initialize_live_state,
    advance_live_tournament,
)


ROUND_LABELS = {
    "GROUPS": "Group Stage",
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter Finals",
    "SF": "Semi Finals",
    "FINAL_STAGE": "Final Stage",
    "DONE": "Tournament Finished",
}


def _format_prob(value: float) -> str:
    return f"{value * 100:.1f}%"


def _flag(team: str) -> str:
    flags = {
        "Argentina": "🇦🇷",
        "Brazil": "🇧🇷",
        "France": "🇫🇷",
        "England": "🏴",
        "Spain": "🇪🇸",
        "Germany": "🇩🇪",
        "Portugal": "🇵🇹",
        "Netherlands": "🇳🇱",
        "Belgium": "🇧🇪",
        "Croatia": "🇭🇷",
        "Uruguay": "🇺🇾",
        "Mexico": "🇲🇽",
        "United States": "🇺🇸",
        "Canada": "🇨🇦",
        "Japan": "🇯🇵",
        "South Korea": "🇰🇷",
        "Morocco": "🇲🇦",
        "Senegal": "🇸🇳",
        "Switzerland": "🇨🇭",
        "Colombia": "🇨🇴",
        "Ecuador": "🇪🇨",
        "Australia": "🇦🇺",
        "Iran": "🇮🇷",
        "Qatar": "🇶🇦",
        "Saudi Arabia": "🇸🇦",
        "Ghana": "🇬🇭",
        "Tunisia": "🇹🇳",
        "Egypt": "🇪🇬",
        "Turkey": "🇹🇷",
        "Norway": "🇳🇴",
        "Sweden": "🇸🇪",
        "Scotland": "🏴",
        "Czechia": "🇨🇿",
        "Austria": "🇦🇹",
        "Algeria": "🇩🇿",
        "Ivory Coast": "🇨🇮",
        "New Zealand": "🇳🇿",
        "Panama": "🇵🇦",
        "Paraguay": "🇵🇾",
        "South Africa": "🇿🇦",
        "Cape Verde": "🇨🇻",
        "Haiti": "🇭🇹",
        "Jordan": "��🇴",
        "Iraq": "🇮🇶",
        "Uzbekistan": "🇺🇿",
        "DR Congo": "🇨🇩",
        "Bosnia and Herzegovina": "🇧🇦",
        "Curaçao": "🇨🇼",
    }
    return flags.get(team, "⚽")


def _team_label(team: str) -> str:
    return f"{_flag(team)} {team}"


def _show_summary_metrics(state: dict) -> None:
    played = state.get("played_matches", pd.DataFrame())
    knockout = state.get("knockout_results", pd.DataFrame())

    group_matches = 0 if played.empty else len(played)
    ko_matches = 0 if knockout.empty else len(knockout)

    goals = 0
    if not played.empty:
        goals += int(played["pred_goals_a"].sum() + played["pred_goals_b"].sum())
    if not knockout.empty:
        goals += int(knockout["goals_a"].sum() + knockout["goals_b"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Phase", ROUND_LABELS.get(state.get("phase"), state.get("phase")))
    c2.metric("Group Matches", group_matches)
    c3.metric("Knockout Matches", ko_matches)
    c4.metric("Total Goals", goals)
    c5.metric("Champion", state.get("champion") or "TBD")


def _show_match_cards(results: pd.DataFrame, title: str) -> None:
    if results.empty:
        return

    st.header(title)

    for _, row in results.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 3])

            with c1:
                st.markdown(f"### {_team_label(row['team_a'])}")
                if "team_a_win_prob" in row:
                    st.caption(f"Win probability: {_format_prob(row['team_a_win_prob'])}")

            with c2:
                st.markdown(
                    f"""
                    <div style="text-align:center; font-size:30px; font-weight:900;">
                        {row['pred_score']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "draw_prob" in row:
                    st.caption(f"Draw: {_format_prob(row['draw_prob'])}")

            with c3:
                st.markdown(f"### {_team_label(row['team_b'])}")
                if "team_b_win_prob" in row:
                    st.caption(f"Win probability: {_format_prob(row['team_b_win_prob'])}")

            winner = row["winner"]
            if winner == "Draw":
                st.info("Result: Draw")
            else:
                st.success(f"Winner: {_team_label(winner)}")


def _show_group_cards(standings: pd.DataFrame) -> None:
    if standings.empty:
        st.info("No standings yet. Play the first matchday.")
        return

    groups = sorted(standings["group"].unique())

    for i in range(0, len(groups), 3):
        cols = st.columns(3)

        for col, group in zip(cols, groups[i:i + 3]):
            with col:
                gdf = standings[standings["group"] == group].copy()
                gdf = gdf.sort_values("position")

                st.markdown(f"### {group}")

                display = gdf[
                    [
                        "position",
                        "team",
                        "played",
                        "wins",
                        "draws",
                        "losses",
                        "goal_diff",
                        "points",
                    ]
                ].copy()

                display["team"] = display["team"].apply(_team_label)
                display["position"] = display["position"].apply(
                    lambda x: f"🟢 {x}" if x <= 2 else f"🔴 {x}"
                )

                st.dataframe(display, use_container_width=True, hide_index=True)


def _show_r32_fixtures(state: dict) -> None:
    r32 = state.get("r32_fixtures", pd.DataFrame())
    if r32.empty:
        return

    st.header("🧩 Round of 32 Fixtures")
    display = r32[["match_slot", "team_a", "team_b"]].copy()
    display["team_a"] = display["team_a"].apply(_team_label)
    display["team_b"] = display["team_b"].apply(_team_label)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _show_knockout_results(state: dict) -> None:
    ko = state.get("knockout_results", pd.DataFrame())
    if ko.empty:
        return

    st.header("🏟️ Knockout Bracket Results")

    order = ["R32", "R16", "QF", "SF", "THIRD_PLACE", "FINAL"]

    for round_name in order:
        rdf = ko[ko["round"] == round_name].copy()
        if rdf.empty:
            continue

        label = ROUND_LABELS.get(round_name, round_name)
        st.markdown(f"### {label}")

        display = rdf[
            [
                "match_slot",
                "team_a",
                "team_b",
                "pred_score",
                "winner",
            ]
        ].copy()

        display["team_a"] = display["team_a"].apply(_team_label)
        display["team_b"] = display["team_b"].apply(_team_label)
        display["winner"] = display["winner"].apply(_team_label)

        st.dataframe(display, use_container_width=True, hide_index=True)


def _show_champion_screen(state: dict) -> None:
    if not state.get("champion"):
        return

    st.balloons()
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; padding:30px; border-radius:20px; background:linear-gradient(135deg,#111827,#1f2937); color:white;">
            <div style="font-size:30px;">🏆 World Cup 2026 Champion</div>
            <div style="font-size:54px; font-weight:900; margin-top:10px;">
                {_team_label(state['champion'])}
            </div>
            <div style="font-size:24px; margin-top:15px;">
                Runner-up: {_team_label(state['runner_up'])}
            </div>
            <div style="font-size:22px; margin-top:8px;">
                Third place: {_team_label(state['third_place'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_all_played_tables(state: dict) -> None:
    played = state.get("played_matches", pd.DataFrame())
    ko = state.get("knockout_results", pd.DataFrame())

    with st.expander("Full simulated group-stage match log"):
        if played.empty:
            st.info("No group-stage matches yet.")
        else:
            df = played[
                [
                    "matchday",
                    "group",
                    "team_a",
                    "team_b",
                    "pred_score",
                    "winner",
                    "team_a_win_prob",
                    "draw_prob",
                    "team_b_win_prob",
                ]
            ].copy()
            for col in ["team_a_win_prob", "draw_prob", "team_b_win_prob"]:
                df[col] = (df[col] * 100).round(1)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Full simulated knockout match log"):
        if ko.empty:
            st.info("No knockout matches yet.")
        else:
            st.dataframe(ko, use_container_width=True, hide_index=True)


def show_live_tournament(model, group_features: pd.DataFrame) -> None:
    st.title("🎮 Live World Cup Tournament Simulator")
    st.caption(
        "Play the tournament step by step: group matchdays, Round of 32, Round of 16, "
        "Quarter Finals, Semi Finals, Third Place and Final."
    )

    if "live_state" not in st.session_state:
        st.session_state.live_state = initialize_live_state(group_features)

    state = st.session_state.live_state

    with st.sidebar:
        st.markdown("## Live Controls")

        if st.button("🔄 Reset Live Tournament", use_container_width=True):
            st.session_state.live_state = initialize_live_state(group_features)
            st.rerun()

        phase = state.get("phase")
        if phase == "GROUPS":
            label = f"▶ Play Matchday {state.get('current_matchday', 1)}"
        elif phase == "DONE":
            label = "🏆 Tournament Finished"
        else:
            label = f"▶ Play {ROUND_LABELS.get(phase, phase)}"

        if st.button(
            label,
            type="primary",
            use_container_width=True,
            disabled=bool(state.get("done", False)),
        ):
            st.session_state.live_state = advance_live_tournament(model, state)
            st.rerun()

        st.divider()
        st.write("Current phase:")
        st.code(ROUND_LABELS.get(st.session_state.live_state.get("phase"), "Unknown"))

    _show_summary_metrics(state)

    _show_champion_screen(state)

    latest_group = state.get("last_matchday_results", pd.DataFrame())
    latest_ko = state.get("last_knockout_results", pd.DataFrame())

    if not latest_group.empty and state.get("phase") in ["GROUPS", "R32"]:
        matchday = int(latest_group["matchday"].iloc[0])
        _show_match_cards(latest_group, f"Latest Results — Matchday {matchday}")

    if not latest_ko.empty:
        round_name = latest_ko["round"].iloc[0]
        _show_match_cards(latest_ko, f"Latest Knockout Results — {round_name}")

    st.divider()

    st.header("📊 Live Group Standings")
    _show_group_cards(state.get("standings", pd.DataFrame()))

    st.divider()

    _show_r32_fixtures(state)

    st.divider()

    _show_knockout_results(state)

    st.divider()

    _show_all_played_tables(state)
