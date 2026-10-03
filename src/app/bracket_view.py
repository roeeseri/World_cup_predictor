"""Pure knockout bracket rendering and team labels; no persistence or session state."""

from __future__ import annotations

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
    "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Mali": "🇲🇱",
    "Serbia": "🇷🇸", "Poland": "🇵🇱", "Ukraine": "🇺🇦", "Romania": "🇷🇴",
    "Hungary": "🇭🇺", "Slovakia": "🇸🇰", "Greece": "🇬🇷", "Denmark": "🇩🇰",
    "Finland": "🇫🇮", "Iceland": "🇮🇸", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Indonesia": "🇮🇩", "Thailand": "🇹🇭",
}

def _flag(team: str) -> str:
    return _FLAGS.get(team, "⚽")

def _team_label(team: str) -> str:
    return f"{_flag(team)} {team}"

_R16_PARENTS: dict[str, tuple[str, str]] = {
    "R16_01": ("R32_01", "R32_02"), "R16_02": ("R32_03", "R32_04"),
    "R16_03": ("R32_05", "R32_06"), "R16_04": ("R32_07", "R32_08"),
    "R16_05": ("R32_09", "R32_10"), "R16_06": ("R32_11", "R32_12"),
    "R16_07": ("R32_13", "R32_14"), "R16_08": ("R32_15", "R32_16"),
}

_QF_PARENTS: dict[str, tuple[str, str]] = {
    "QF_01": ("R16_01", "R16_02"), "QF_02": ("R16_03", "R16_04"),
    "QF_03": ("R16_05", "R16_06"), "QF_04": ("R16_07", "R16_08"),
}

_SF_PARENTS: dict[str, tuple[str, str]] = {
    "SF_01": ("QF_01", "QF_02"), "SF_02": ("QF_03", "QF_04"),
}

def _resolve_winner(parent_slot: str, ko_results: dict, r32_lookup: dict) -> str:
    """Return the team that won *parent_slot*, or a descriptive placeholder."""
    if parent_slot in ko_results:
        w = ko_results[parent_slot].get("winner")
        if w:
            return w
        # Result stored but winner field missing (old record without penalty info)
        r = ko_results[parent_slot]
        ga, gb = r["goals_a"], r["goals_b"]
        if ga > gb:
            return r.get("team_a", f"W of {parent_slot.replace('_','-')}")
        if gb > ga:
            return r.get("team_b", f"W of {parent_slot.replace('_','-')}")
    # Not yet played
    if parent_slot.startswith("R32_"):
        num = int(parent_slot[-2:])
        if parent_slot in r32_lookup:
            ta, tb = r32_lookup[parent_slot]
            return f"W of M{num} ({ta} vs {tb})"
        return f"W of M{num}"
    return f"W of {parent_slot.replace('_', '-')}"

def _resolve_loser(parent_slot: str, ko_results: dict) -> str:
    """Return the team that lost *parent_slot*, or a placeholder."""
    if parent_slot in ko_results:
        r = ko_results[parent_slot]
        w = r.get("winner")
        ta, tb = r.get("team_a", ""), r.get("team_b", "")
        if w == ta and tb:
            return tb
        if w == tb and ta:
            return ta
    return f"L of {parent_slot.replace('_', '-')}"

def _resolve_ko_match_teams(
    slot: str,
    ko_results: dict,
    r32_lookup: dict,
) -> tuple[str, str]:
    """Resolve the two teams for any knockout slot using known results where available."""
    if slot.startswith("R32_"):
        return r32_lookup.get(slot, ("TBD", "TBD"))
    if slot in _R16_PARENTS:
        pa, pb = _R16_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot in _QF_PARENTS:
        pa, pb = _QF_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot in _SF_PARENTS:
        pa, pb = _SF_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot == "FINAL":
        return _resolve_winner("SF_01", ko_results, r32_lookup), _resolve_winner("SF_02", ko_results, r32_lookup)
    if slot == "THIRD_PLACE":
        return _resolve_loser("SF_01", ko_results), _resolve_loser("SF_02", ko_results)
    return ("TBD", "TBD")

def _is_real_team(name: str) -> bool:
    """True if name is an actual team, not a placeholder."""
    return bool(name) and not name.startswith(("W of", "L of", "Winner", "Loser", "TBD"))

def _build_bracket_html(r32_teams: dict[str, tuple[str, str]], ko_results: dict | None = None) -> str:
    """Return HTML string for the full knockout bracket visualization."""
    CARD_W = 140
    CARD_H = 56
    COL_PITCH = 155
    UNIT = 130
    HDR_H = 24

    xs = [i * COL_PITCH for i in range(9)]  # 0,155,310,465,620,775,930,1085,1240
    TOTAL_W = xs[8] + CARD_W  # 1380
    TOTAL_H = HDR_H + 8 * UNIT  # 1064

    def _cy(slot_idx: int) -> int:
        return HDR_H + UNIT * slot_idx + UNIT // 2

    r32_cy = [_cy(i) for i in range(8)]
    r16_cy = [(r32_cy[i * 2] + r32_cy[i * 2 + 1]) // 2 for i in range(4)]
    qf_cy  = [(r16_cy[i * 2] + r16_cy[i * 2 + 1]) // 2 for i in range(2)]
    sf_cy  = (qf_cy[0] + qf_cy[1]) // 2

    LINE_C   = "#3d4166"
    CARD_BG  = "#1a1d27"
    BORD_C   = "#2d3142"
    TEXT_C   = "#d4d4d8"
    DIM_C    = "#6b6b8a"
    HDR_BG   = "#14162a"
    FIN_BG   = "#1e1a2e"
    FIN_BORD = "#5b4fcf"

    def _esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _card(x: int, center_y: int, ta: str, tb: str, label: str, is_final: bool = False) -> str:
        top  = center_y - CARD_H // 2
        bg   = FIN_BG   if is_final else CARD_BG
        bord = FIN_BORD if is_final else BORD_C
        return (
            f'<div style="position:absolute;left:{x}px;top:{top}px;width:{CARD_W}px;height:{CARD_H}px;'
            f'border:1px solid {bord};border-radius:4px;background:{bg};overflow:hidden;">'
            f'<div style="padding:1px 5px;height:21px;line-height:21px;font-size:10px;'
            f'color:{TEXT_C};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'border-bottom:1px solid #22243a;">{_esc(ta)}</div>'
            f'<div style="padding:1px 5px;height:21px;line-height:21px;font-size:10px;'
            f'color:{TEXT_C};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_esc(tb)}</div>'
            f'<div style="height:12px;line-height:12px;font-size:8px;color:{DIM_C};'
            f'background:{HDR_BG};padding:0 4px;text-align:center;">{_esc(label)}</div>'
            f'</div>'
        )

    def _hl(x: int, y: int, w: int) -> str:
        if w <= 0:
            return ""
        return (f'<div style="position:absolute;left:{x}px;top:{y}px;'
                f'width:{w}px;height:1px;background:{LINE_C};"></div>')

    def _vl(x: int, y1: int, y2: int) -> str:
        if y2 <= y1:
            return ""
        return (f'<div style="position:absolute;left:{x}px;top:{y1}px;'
                f'width:1px;height:{y2 - y1}px;background:{LINE_C};"></div>')

    def _lconn(col_x: int, y1: int, y2: int) -> str:
        """Left-to-right: pair of cards at col_x feeds into next column."""
        mid   = (y1 + y2) // 2
        re    = col_x + CARD_W
        vx    = re + 7
        nl    = col_x + COL_PITCH
        return _hl(re, y1, vx - re) + _hl(re, y2, vx - re) + _vl(vx, min(y1, y2), max(y1, y2)) + _hl(vx, mid, nl - vx)

    def _rconn(inner_x: int, outer_x: int, y1: int, y2: int) -> str:
        """Right-to-left: pair of cards at outer_x feeds into inner column."""
        mid   = (y1 + y2) // 2
        ir    = inner_x + CARD_W
        vx    = ir + 7
        return _hl(vx, y1, outer_x - vx) + _hl(vx, y2, outer_x - vx) + _vl(vx, min(y1, y2), max(y1, y2)) + _hl(ir, mid, vx - ir)

    _ko = ko_results or {}

    def _bteam(slot: str) -> tuple[str, str]:
        """Resolve bracket card teams for any slot, using known results where available."""
        ta, tb = _resolve_ko_match_teams(slot, _ko, r32_teams)
        def _fmt(t: str) -> str:
            if _is_real_team(t):
                return f"{_flag(t)} {t}"
            return t
        return _fmt(ta), _fmt(tb)

    parts: list[str] = []

    # Round header labels
    hdr_labels = ["Rd of 32", "Rd of 16", "Qtr Finals", "Semi Finals",
                  "🏆 FINAL", "Semi Finals", "Qtr Finals", "Rd of 16", "Rd of 32"]
    for i, hl in enumerate(hdr_labels):
        parts.append(
            f'<div style="position:absolute;left:{xs[i]}px;top:0;width:{CARD_W}px;height:{HDR_H}px;'
            f'text-align:center;font-size:9px;color:{DIM_C};line-height:{HDR_H}px;'
            f'text-transform:uppercase;letter-spacing:0.4px;">{hl}</div>'
        )

    # ── Left R32 ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["R32_01","R32_02","R32_03","R32_04","R32_05","R32_06","R32_07","R32_08"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[0], r32_cy[i], ta_lbl, tb_lbl, f"M{int(slot[-2:])}"))
    for i in range(4):
        parts.append(_lconn(xs[0], r32_cy[i * 2], r32_cy[i * 2 + 1]))

    # ── Left R16 ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["R16_01","R16_02","R16_03","R16_04"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[1], r16_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    for i in range(2):
        parts.append(_lconn(xs[1], r16_cy[i * 2], r16_cy[i * 2 + 1]))

    # ── Left QF ─────────────────────────────────────────────────────────────
    for i, slot in enumerate(["QF_01","QF_02"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[2], qf_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    parts.append(_lconn(xs[2], qf_cy[0], qf_cy[1]))

    # ── Left SF ─────────────────────────────────────────────────────────────
    sf1_ta, sf1_tb = _bteam("SF_01")
    parts.append(_card(xs[3], sf_cy, sf1_ta, sf1_tb, "SF-01"))
    parts.append(_hl(xs[3] + CARD_W, sf_cy, xs[4] - (xs[3] + CARD_W)))

    # ── Final ───────────────────────────────────────────────────────────────
    fin_ta, fin_tb = _bteam("FINAL")
    parts.append(_card(xs[4], sf_cy, fin_ta, fin_tb, "🏆 FINAL", is_final=True))
    parts.append(_hl(xs[4] + CARD_W, sf_cy, xs[5] - (xs[4] + CARD_W)))

    # ── Right SF ────────────────────────────────────────────────────────────
    sf2_ta, sf2_tb = _bteam("SF_02")
    parts.append(_card(xs[5], sf_cy, sf2_ta, sf2_tb, "SF-02"))
    parts.append(_rconn(xs[5], xs[6], qf_cy[0], qf_cy[1]))

    # ── Right QF ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["QF_03","QF_04"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[6], qf_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    parts.append(_rconn(xs[6], xs[7], r16_cy[0], r16_cy[1]))
    parts.append(_rconn(xs[6], xs[7], r16_cy[2], r16_cy[3]))

    # ── Right R16 ───────────────────────────────────────────────────────────
    for i, slot in enumerate(["R16_05","R16_06","R16_07","R16_08"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[7], r16_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    for i in range(4):
        parts.append(_rconn(xs[7], xs[8], r32_cy[i * 2], r32_cy[i * 2 + 1]))

    # ── Right R32 ───────────────────────────────────────────────────────────
    for i, slot in enumerate(["R32_09","R32_10","R32_11","R32_12","R32_13","R32_14","R32_15","R32_16"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[8], r32_cy[i], ta_lbl, tb_lbl, f"M{int(slot[-2:])}"))

    return (
        f'<div style="width:{TOTAL_W}px;height:{TOTAL_H}px;position:relative;'
        f'background:#0d0f1a;border-radius:8px;font-family:system-ui,sans-serif;">'
        + "".join(parts)
        + "</div>"
    )
