"""Scrape Transfermarkt squad value by position for national teams across years."""

from pathlib import Path
import sys
import time

import pandas as pd
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data/processed/transfermarkt_market_values_clean.csv"

PLAYER_OUT = PROJECT_ROOT / "data/raw/transfermarkt/transfermarkt_player_position_values_2004_2026.csv"
TEAM_OUT = PROJECT_ROOT / "data/processed/transfermarkt_position_values_2004_2026.csv"

START_YEAR = 2004
END_YEAR = 2026


EXTRACT_JS = """
() => {
  function parseMarketValue(valueText) {
    if (!valueText) return 0;

    const clean = valueText
      .replace("€", "")
      .replace(",", "")
      .trim()
      .toLowerCase();

    if (clean.includes("bn")) return parseFloat(clean) * 1000;
    if (clean.includes("m")) return parseFloat(clean);
    if (clean.includes("k")) return parseFloat(clean) / 1000;

    return parseFloat(clean) || 0;
  }

  function positionGroup(position) {
    const p = position.toLowerCase();

    if (p.includes("goalkeeper")) return "Goalkeeper";
    if (p.includes("back") || p.includes("defender") || p.includes("sweeper")) return "Defender";
    if (p.includes("midfield") || p.includes("midfielder")) return "Midfield";
    if (p.includes("winger") || p.includes("forward") || p.includes("striker")) return "Attack";

    return "Other";
  }

  const rows = [...document.querySelectorAll("table.items tbody tr")];

  return rows.map(row => {
    const cells = [...row.querySelectorAll("td")];

    if (cells.length < 8) return null;

    const name = cells[3]?.innerText?.trim() || "";
    const position = cells[4]?.innerText?.trim() || "";
    const ageText = cells[5]?.innerText?.trim() || "";
    const marketText = cells[7]?.innerText?.trim() || "";

    const age = ageText ? Number(ageText) : null;
    const marketValueMillions = parseMarketValue(marketText);

    return {
      name,
      position,
      position_group: positionGroup(position),
      age,
      market_value_millions_eur: marketValueMillions
    };
  }).filter(p => p && p.name);
}
"""


def summarize_team(players: list[dict]) -> dict:
    summary = {}

    for group in ["Goalkeeper", "Defender", "Midfield", "Attack", "Other"]:
        group_players = [p for p in players if p["position_group"] == group]

        group_key = group.lower()

        summary[f"{group_key}_players"] = len(group_players)
        summary[f"{group_key}_market_value_millions_eur"] = sum(
            p["market_value_millions_eur"] for p in group_players
        )

        ages = [p["age"] for p in group_players if p["age"] is not None]
        summary[f"{group_key}_avg_age"] = sum(ages) / len(ages) if ages else 0

    total_value = sum(p["market_value_millions_eur"] for p in players)
    total_players = len(players)
    ages = [p["age"] for p in players if p["age"] is not None]

    summary["scraped_players"] = total_players
    summary["scraped_total_market_value_millions_eur"] = total_value
    summary["scraped_avg_age"] = sum(ages) / len(ages) if ages else 0

    return summary


def make_year_url(url: str, season_id: int) -> str:
    if "saison_id=" in url:
        base = url.split("saison_id=")[0]
        return f"{base}saison_id={season_id}"

    if "/saison_id/" in url:
        parts = url.split("/saison_id/")
        prefix = parts[0]
        return f"{prefix}/saison_id/{season_id}"

    separator = "&" if "?" in url else "?"
    return f"{url}{separator}saison_id={season_id}"


def load_existing_keys() -> set[tuple[str, int]]:
    if not TEAM_OUT.exists():
        return set()

    existing = pd.read_csv(TEAM_OUT)

    if existing.empty:
        return set()

    return set(zip(existing["team_name_tm"], existing["season_id"]))


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0

    df.to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    tm = pd.read_csv(INPUT_PATH)

    teams = (
        tm[
            (tm["season_id"].between(START_YEAR, END_YEAR)) &
            (tm["url"].notna())
        ][["team_name_tm", "season_id", "url"]]
        .drop_duplicates()
        .sort_values(["season_id", "team_name_tm"])
        .reset_index(drop=True)
    )

    existing_keys = load_existing_keys()

    teams = teams[
        ~teams.apply(
            lambda row: (row["team_name_tm"], int(row["season_id"])) in existing_keys,
            axis=1,
        )
    ].reset_index(drop=True)

    print("Remaining team-year pages to scrape:", len(teams))
    print("Output team summary:", TEAM_OUT)
    print("Output players:", PLAYER_OUT)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        )

        for i, row in teams.iterrows():
            team = row["team_name_tm"]
            season_id = int(row["season_id"])
            url = make_year_url(row["url"], season_id)

            print(f"[{i + 1}/{len(teams)}] {season_id} - {team}")

            players_rows = []
            team_rows = []

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(1.2)

                players = page.evaluate(EXTRACT_JS)

                for player in players:
                    player["team_name_tm"] = team
                    player["season_id"] = season_id
                    player["source_url"] = page.url

                team_summary = summarize_team(players)
                team_summary["team_name_tm"] = team
                team_summary["season_id"] = season_id
                team_summary["source_url"] = page.url
                team_summary["scrape_status"] = "ok"

                players_rows.extend(players)
                team_rows.append(team_summary)

            except Exception as e:
                print("FAILED:", season_id, team, e)

                team_rows.append({
                    "team_name_tm": team,
                    "season_id": season_id,
                    "source_url": url,
                    "scrape_status": f"failed: {e}",
                    "scraped_players": 0,
                    "scraped_total_market_value_millions_eur": 0,
                    "scraped_avg_age": 0,
                })

            append_csv(PLAYER_OUT, players_rows)
            append_csv(TEAM_OUT, team_rows)

            time.sleep(0.8)

        browser.close()

    print("Done.")


if __name__ == "__main__":
    main()
