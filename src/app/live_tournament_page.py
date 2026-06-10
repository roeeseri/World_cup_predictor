"""Live tournament page for WC 2026 — real results, standings, simulate forward."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.state.live_state import (
    apply_results_from_csv,
    initialize_live_state,
    record_match_result,
    simulate_forward,
)
from src.tournament.group_standings import build_group_standings

UPDATES_CSV = Path("data/raw/world_cup_updates/all_world_cup_2026_updates.csv")

ROUND_LABELS = {
    "GROUPS": "Group Stage",
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter Finals",
    "SF": "Semi Finals",
    "FINAL_STAGE": "Final Stage",
    "DONE": "Tournament Finished",
}

_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Mexico": "🇲🇽",
    "United States": "🇺🇸", "USA": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Morocco": "🇲🇦",
    "Senegal": "🇸🇳", "Switzerland": "🇨🇭", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Australia": "🇦🇺", "Iran": "🇮🇷", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Ghana": "🇬🇭", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Turkey": "🇹🇷", "Norway": "🇳🇴", "Sweden": "🇸🇪",
    "Czechia": "🇨🇿", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Ivory Coast": "🇨🇮", "New Zealand": "🇳🇿", "Panama": "🇵🇦",
    "Paraguay": "🇵🇾", "South Africa": "🇿🇦", "Cape Verde": "🇨🇻",
    "Haiti": "🇭🇹", "Jordan": "🇯🇴", "Iraq": "🇮🇶", "Uzbekistan": "🇺🇿",
    "DR Congo": "🇨🇩", "Bosnia and Herzegovina": "🇧🇦", "Curaçao": "🇨🇼",
    "Venezuela": "🇻🇪", "Chile": "🇨🇱", "Peru": "🇵🇪", "Bolivia": "🇧🇴",
    "Costa Rica": "🇨🇷", "Honduras": "🇭🇳", "Jamaica": "🇯🇲",
    "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Ghana": "🇬🇭", "Mali": "🇲🇱",
    "Serbia": "🇷🇸", "Poland": "🇵🇱", "Ukraine": "🇺🇦", "Romania": "🇷🇴",
    "Hungary": "🇭🇺", "Slovakia": "🇸🇰", "Greece": "🇬🇷", "Denmark": "🇩🇰",
    "Finland": "🇫🇮", "Iceland": "🇮🇸", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "New Zealand": "🇳🇿", "Indonesia": "🇮🇩", "Thailand": "🇹🇭",
}


def _flag(team: str) -> str:
    return _FLAGS.get(team, "⚽")


def _team_label(team: str) -> str:
    return f"{_flag(team)} {team}"


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state(historical_matches: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    if "true_state" not in st.session_state:
        state = initialize_live_state(historical_matches, fixtures)
        state = apply_results_from_csv(state, UPDATES_CSV)
        st.session_state.true_state = state
        st.session_state.sim_state = None


def _refresh_from_csv() -> None:
    st.session_state.true_state = apply_results_from_csv(
        st.session_state.true_state, UPDATES_CSV
    )
    st.session_state.sim_state = None


def _submit_result(match_id: int, goals_a: int, goals_b: int) -> None:
    st.session_state.true_state = record_match_result(
        st.session_state.true_state, match_id, goals_a, goals_b
    )
    st.session_state.sim_state = None


# ---------------------------------------------------------------------------
# Match prediction helper
# ---------------------------------------------------------------------------

def _get_prediction(
    model,
    state: dict,
    team_a: str,
    team_b: str,
    match_date,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
) -> dict | None:
    from src.features.build_features import build_pre_match_features
    from src.features.team_names import normalize_team_name
    from src.models.score_conversion import most_likely_score, win_draw_loss_probs

    if feature_fn is None:
        feature_fn = build_pre_match_features
    if score_fn is None:
        score_fn = most_likely_score

    try:
        feature_row = feature_fn(
            team_a=normalize_team_name(team_a),
            team_b=normalize_team_name(team_b),
            match_date=match_date,
            team_states=state["team_states"],
            historical_matches=state["historical_matches"],
            market_values=market_values,
            position_values=position_values,
            elo_ratings=state["elo_ratings"],
            rankings=state["rankings"],
        )
        pred = model.predict(feature_row.fillna(0))
        la = float(pred[0, 0])
        lb = float(pred[0, 1])
        ga, gb = score_fn(la, lb)
        win_a, draw, win_b = win_draw_loss_probs(la, lb)
        return {
            "lambda_a": la, "lambda_b": lb,
            "pred_goals_a": ga, "pred_goals_b": gb,
            "win_a": win_a, "draw": draw, "win_b": win_b,
        }
    except Exception as e:
        # Surface the error so it's debuggable from the UI
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Match card rendering
# ---------------------------------------------------------------------------

def _render_completed_match(fixture: pd.Series) -> None:
    ga = int(fixture["goals_a"])
    gb = int(fixture["goals_b"])
    ta, tb = fixture["team_a"], fixture["team_b"]
    time_str = pd.to_datetime(fixture["date"], utc=True).strftime("%H:%M UTC")

    if ga > gb:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  ✅ {ta} wins"
    elif gb > ga:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  ✅ {tb} wins"
    else:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  🤝 Draw"

    st.success(f"🕐 {time_str}  |  {result}")


def _render_upcoming_match(
    fixture: pd.Series,
    model,
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
) -> None:
    ta, tb = fixture["team_a"], fixture["team_b"]
    mid = int(fixture["match_id"])
    match_date = fixture["date"]
    time_str = pd.to_datetime(match_date, utc=True).strftime("%H:%M UTC")

    pred = _get_prediction(model, state, ta, tb, match_date, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn)

    with st.container(border=True):
        header_col, prob_col = st.columns([2, 3])

        with header_col:
            st.markdown(f"**🕐 {time_str}**")
            st.markdown(f"{_team_label(ta)}  vs  {_team_label(tb)}")
            if pred and "_error" not in pred:
                st.markdown(
                    f"Model prediction: **{pred['pred_goals_a']} – {pred['pred_goals_b']}**"
                    f"  *(xG {pred['lambda_a']:.2f} – {pred['lambda_b']:.2f})*"
                )
            elif pred and "_error" in pred:
                st.caption(f"⚠️ Prediction error: {pred['_error']}")
            else:
                st.caption("*Prediction unavailable*")

        with prob_col:
            if pred and "_error" not in pred:
                c1, c2, c3 = st.columns(3)
                c1.metric(_team_label(ta), f"{pred['win_a'] * 100:.0f}%")
                c2.metric("Draw", f"{pred['draw'] * 100:.0f}%")
                c3.metric(_team_label(tb), f"{pred['win_b'] * 100:.0f}%")

        with st.expander("Enter actual result"):
            fc1, fc2, fc3 = st.columns([2, 1, 2])
            with fc1:
                ga_input = st.number_input(
                    f"{ta} goals", min_value=0, max_value=20, value=0,
                    key=f"ga_{mid}",
                )
            with fc2:
                st.markdown("<br><div style='text-align:center'>–</div>",
                            unsafe_allow_html=True)
            with fc3:
                gb_input = st.number_input(
                    f"{tb} goals", min_value=0, max_value=20, value=0,
                    key=f"gb_{mid}",
                )
            if st.button("✅ Submit result", key=f"submit_{mid}"):
                _submit_result(mid, int(ga_input), int(gb_input))
                st.rerun()


# ---------------------------------------------------------------------------
# Fixtures by day tab
# ---------------------------------------------------------------------------

def _show_group_stage(
    state: dict,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
) -> None:
    fixtures = state["fixtures"].copy()

    fixtures["_date_label"] = pd.to_datetime(
        fixtures["date"], utc=True
    ).dt.strftime("%b %d")
    fixtures["_date_sort"] = pd.to_datetime(fixtures["date"], utc=True).dt.date

    dates_df = (
        fixtures[["_date_label", "_date_sort"]]
        .drop_duplicates()
        .sort_values("_date_sort")
    )
    unique_date_labels = dates_df["_date_label"].tolist()

    if not unique_date_labels:
        st.info("No fixtures loaded.")
        return

    selected_date = st.selectbox("Select match day", unique_date_labels, key="day_selector")
    day_fixtures = fixtures[fixtures["_date_label"] == selected_date].sort_values("date")

    # Group / matchday banner
    groups_today = sorted(day_fixtures["group"].unique())
    matchdays_today = sorted(day_fixtures["matchday"].unique())
    st.markdown(
        f"### {selected_date}  —  Matchday {', '.join(str(m) for m in matchdays_today)}  "
        f"·  Groups: {', '.join(groups_today)}"
    )

    for _, fix in day_fixtures.iterrows():
        if bool(fix.get("is_completed", False)):
            _render_completed_match(fix)
        else:
            _render_upcoming_match(fix, model, state, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn)

    # Day summary
    day_completed = day_fixtures[day_fixtures["is_completed"]]
    if not day_completed.empty:
        goals_today = int(day_completed["goals_a"].sum() + day_completed["goals_b"].sum())
        st.caption(f"{len(day_completed)} result(s) recorded today · {goals_today} goals")


# ---------------------------------------------------------------------------
# Group standings tab
# ---------------------------------------------------------------------------

def _build_full_standings(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build standings for ALL groups, including teams with 0 games played."""
    completed = fixtures[fixtures["is_completed"]].copy()

    # Start from completed matches (may be empty)
    if not completed.empty:
        completed["goals_a"] = completed["goals_a"].astype(int)
        completed["goals_b"] = completed["goals_b"].astype(int)
        standings = build_group_standings(
            completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
        )
    else:
        standings = pd.DataFrame(
            columns=["group", "position", "team", "played", "wins", "draws",
                     "losses", "goals_for", "goals_against", "goal_diff", "points"]
        )

    # Collect every team from all fixtures
    all_teams = pd.concat([
        fixtures[["group", "team_a"]].rename(columns={"team_a": "team"}),
        fixtures[["group", "team_b"]].rename(columns={"team_b": "team"}),
    ]).drop_duplicates()

    played_teams = set(standings["team"]) if not standings.empty else set()
    zero_rows = []
    for _, row in all_teams.iterrows():
        if row["team"] not in played_teams:
            zero_rows.append({
                "group": row["group"], "team": row["team"],
                "position": 0, "played": 0, "wins": 0, "draws": 0,
                "losses": 0, "goals_for": 0, "goals_against": 0,
                "goal_diff": 0, "points": 0,
            })

    if zero_rows:
        standings = pd.concat(
            [standings, pd.DataFrame(zero_rows)], ignore_index=True
        )

    # Re-sort: points desc → goal_diff desc → goals_for desc → team asc
    standings = standings.sort_values(
        ["group", "points", "goal_diff", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    # Re-assign position within each group
    standings["position"] = standings.groupby("group").cumcount() + 1

    return standings


def _show_standings(state: dict) -> None:
    fixtures = state["fixtures"]
    standings = _build_full_standings(fixtures)
    completed = fixtures[fixtures["is_completed"]].copy()

    # Standings tables — always show all groups
    groups_all = sorted(standings["group"].unique())
    for i in range(0, len(groups_all), 3):
        cols = st.columns(3)
        for col, grp in zip(cols, groups_all[i : i + 3]):
            with col:
                st.markdown(f"**{grp}**")
                gdf = standings[standings["group"] == grp][
                    ["position", "team", "played", "wins", "draws",
                     "losses", "goals_for", "goals_against", "goal_diff", "points"]
                ].copy()
                gdf["team"] = gdf["team"].apply(_team_label)
                gdf["position"] = gdf["position"].apply(
                    lambda x: f"🟢 {x}" if x <= 2 else f"⚪ {x}"
                )
                st.dataframe(gdf, use_container_width=True, hide_index=True)

    st.divider()

    # Goals per team + ELO table (only meaningful once some games played)
    col_goals, col_elo = st.columns(2)

    with col_goals:
        st.markdown("### Goals scored per team")
        if completed.empty:
            st.caption("No goals yet.")
        else:
            completed["goals_a"] = completed["goals_a"].astype(int)
            completed["goals_b"] = completed["goals_b"].astype(int)
            goals_a = completed.groupby("team_a")["goals_a"].sum().rename("goals")
            goals_b = completed.groupby("team_b")["goals_b"].sum().rename("goals")
            team_goals = (
                pd.concat([goals_a, goals_b])
                .groupby(level=0).sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            team_goals.columns = ["team", "goals"]
            team_goals["team"] = team_goals["team"].apply(_team_label)
            st.dataframe(team_goals, use_container_width=True, hide_index=True)

    with col_elo:
        st.markdown("### Current ELO ratings (with tournament change)")
        hist = state["historical_matches"]
        # Baseline = last ELO before any WC 2026 live row
        pre = (
            hist[hist["tournament_key"] != "FIFA World Cup_2026"]
            if "tournament_key" in hist.columns
            else hist[hist["source_file"] != "live_2026"]
            if "source_file" in hist.columns
            else hist
        )
        baseline: dict[str, float] = {}
        for _, row in pre.sort_values("date").iterrows():
            baseline[row["team_a"]] = float(row["rating_a"])
            baseline[row["team_b"]] = float(row["rating_b"])

        elo_rows = []
        for team, current in state["elo_ratings"].items():
            if team in baseline:
                delta = current - baseline[team]
                elo_rows.append({
                    "team": _team_label(team),
                    "ELO": round(current, 1),
                    "Δ tournament": round(delta, 1),
                })
        if elo_rows:
            elo_df = (
                pd.DataFrame(elo_rows)
                .sort_values("ELO", ascending=False)  # highest rating first
                .reset_index(drop=True)
            )
            st.dataframe(elo_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Simulate forward tab
# ---------------------------------------------------------------------------

def _show_simulate_forward(
    state: dict,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    unplayed = state["fixtures"][~state["fixtures"]["is_completed"]]
    n = len(unplayed)

    completed_count = int(state["fixtures"]["is_completed"].sum())
    total_count = len(state["fixtures"])
    st.markdown(
        f"**{completed_count}/{total_count}** group-stage matches have real results.  "
        f"**{n}** match(es) will be simulated using the v3 model."
    )

    if n == 0:
        st.success("All group stage matches have been played!")
    else:
        if st.button("🎲 Simulate remaining group matches", type="primary"):
            with st.spinner(f"Simulating {n} match(es)…"):
                st.session_state.sim_state = simulate_forward(
                    state, model, market_values, position_values,
                    feature_fn=feature_fn, score_fn=score_fn,
                )

    sim = st.session_state.get("sim_state")
    if sim is None:
        return

    st.warning("⚠️ SIMULATION — predicted scores, not actual results")

    sim_fixtures = sim["fixtures"]
    sim_completed = sim_fixtures[sim_fixtures["is_completed"]].copy()

    real_ids = set(
        state["fixtures"][state["fixtures"]["is_completed"]]["match_id"].tolist()
    )
    sim_only = sim_completed[~sim_completed["match_id"].isin(real_ids)].copy()

    if not sim_only.empty:
        st.markdown("### Simulated results")
        sim_only["goals_a"] = sim_only["goals_a"].astype(int)
        sim_only["goals_b"] = sim_only["goals_b"].astype(int)
        for _, fix in sim_only.sort_values("date").iterrows():
            ga, gb = int(fix["goals_a"]), int(fix["goals_b"])
            ta, tb = fix["team_a"], fix["team_b"]
            st.markdown(
                f"- {_team_label(ta)} **{ga}–{gb}** {_team_label(tb)}"
            )

    # Simulated standings
    st.markdown("### Simulated group standings")
    sim_completed["goals_a"] = sim_completed["goals_a"].astype(int)
    sim_completed["goals_b"] = sim_completed["goals_b"].astype(int)

    sim_standings = build_group_standings(
        sim_completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
    )
    groups_all = sorted(sim_standings["group"].unique())
    for i in range(0, len(groups_all), 3):
        cols = st.columns(3)
        for col, grp in zip(cols, groups_all[i : i + 3]):
            with col:
                st.markdown(f"**{grp}**")
                gdf = sim_standings[sim_standings["group"] == grp][
                    ["position", "team", "played", "wins", "draws", "losses",
                     "goals_for", "goals_against", "goal_diff", "points"]
                ].copy()
                gdf["team"] = gdf["team"].apply(_team_label)
                gdf["position"] = gdf["position"].apply(
                    lambda x: f"🟢 {x}" if x <= 2 else f"⚪ {x}"
                )
                st.dataframe(gdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def show_live_tournament(
    model,
    fixtures: pd.DataFrame,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    """Render the full Live Tournament page."""
    _init_state(historical_matches, fixtures)

    state = st.session_state.true_state

    st.header("⚽ WC 2026 — Live Tournament")

    # Header bar
    completed_count = int(state["fixtures"]["is_completed"].sum())
    total_count = len(state["fixtures"])
    hdr_col, btn_col, reset_col = st.columns([4, 1, 1])
    with hdr_col:
        st.caption(f"{completed_count}/{total_count} group stage matches with real results")
    with btn_col:
        if st.button("🔄 Refresh from CSV"):
            _refresh_from_csv()
            st.rerun()
    with reset_col:
        if st.button("🗑️ Reset state"):
            for key in ("true_state", "sim_state"):
                st.session_state.pop(key, None)
            st.rerun()

    tab_gs, tab_standings, tab_sim = st.tabs(
        ["📅 Fixtures by Day", "📊 Group Standings", "🎲 Simulate Forward"]
    )

    with tab_gs:
        _show_group_stage(state, model, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn)

    with tab_standings:
        _show_standings(state)

    with tab_sim:
        _show_simulate_forward(state, model, market_values, position_values, score_fn=score_fn, feature_fn=feature_fn)
