"""Knockout presentation must follow recorded winners and escape team labels."""
from src.app.bracket_view import _build_bracket_html, _resolve_ko_match_teams


def test_final_and_third_place_follow_semifinal_results():
    results = {
        "SF_01": {"team_a": "Argentina", "team_b": "France", "winner": "France"},
        "SF_02": {"team_a": "Brazil", "team_b": "Japan", "winner": "Brazil"},
    }
    assert _resolve_ko_match_teams("FINAL", results, {}) == ("France", "Brazil")
    assert _resolve_ko_match_teams("THIRD_PLACE", results, {}) == ("Argentina", "Japan")


def test_unplayed_round_keeps_placeholders():
    teams = _resolve_ko_match_teams("R16_01", {}, {"R32_01": ("Argentina", "France")})
    assert "Argentina vs France" in teams[0]
    assert teams[1].startswith("W of")


def test_bracket_html_escapes_team_names():
    html = _build_bracket_html({"R32_01": ("<Test & Team>", "France")})
    assert "<Test & Team>" not in html
    assert "&lt;Test &amp; Team&gt;" in html
