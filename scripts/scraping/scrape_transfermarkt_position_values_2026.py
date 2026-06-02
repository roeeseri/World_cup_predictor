"""Scrape Transfermarkt squad value by position for national teams."""

from pathlib import Path
import time

import pandas as pd
from playwright.sync_api import sync_playwright


INPUT_PATH = Path("data/processed/transfermarkt_market_values_clean.csv")
PLAYER_OUT = Path("data/raw/transfermarkt/transfermarkt_player_position_values_2026.csv")
TEAM_OUT = Path("data/processed/transfermarkt_position_values_2026.csv")


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

    if (
      p.includes("back") ||
      p.includes("defender") ||
      p.includes("sweeper")
    ) return "Defender";

    if (
      p.includes("midfield") ||
      p.includes("midfielder")
    ) return "Midfield";

    if (
      p.includes("winger") ||
      p.includes("forward") ||
      p.includes("striker")
    ) return "Attack";

    return "Other";
  }

  const rows = [...document.querySelectorAll("table.items tbody tr")];

  return rows.map(row => {
    const cells = [...row.querySelectorAll("td")];

    // Transfermarkt detailed squad uses one real player row with 8 cells,
    // followed by helper rows with 1-2 cells. Keep only real rows.
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

        summary[f"{group.lower()}_players"] = len(group_players)
        summary[f"{group.lower()}_market_value_millions_eur"] = sum(
            p["market_value_millions_eur"] for p in group_players
        )

        ages = [p["age"] for p in group_players if p["age"] is not None]
        summary[f"{group.lower()}_avg_age"] = sum(ages) / len(ages) if ages else 0

    total_value = sum(p["market_value_millions_eur"] for p in players)
    total_players = len(players)
    ages = [p["age"] for p in players if p["age"] is not None]

    summary["scraped_players"] = total_players
    summary["scraped_total_market_value_millions_eur"] = total_value
    summary["scraped_avg_age"] = sum(ages) / len(ages) if ages else 0

    return summary


def main() -> None:
    tm = pd.read_csv(INPUT_PATH)

    season_id = 2026
    teams = (
        tm[tm["season_id"] == season_id]
        [["team_name_tm", "season_id", "url"]]
        .drop_duplicates()
        .dropna(subset=["url"])
        .sort_values("team_name_tm")
        .reset_index(drop=True)
    )

    print("Teams to scrape:", len(teams))

    all_players = []
    all_team_rows = []

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
            url = row["url"]

            print(f"[{i + 1}/{len(teams)}] {team}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(1.5)

                detailed_links = page.locator("a", has_text="Detailed squad")

                if detailed_links.count() > 0:
                    href = detailed_links.first.get_attribute("href")
                    if href:
                        if href.startswith("/"):
                            href = "https://www.transfermarkt.com" + href
                        page.goto(href, wait_until="domcontentloaded", timeout=60000)
                        time.sleep(1.5)

                players = page.evaluate(EXTRACT_JS)

                for player in players:
                    player["team_name_tm"] = team
                    player["season_id"] = season_id
                    player["source_url"] = page.url

                team_summary = summarize_team(players)
                team_summary["team_name_tm"] = team
                team_summary["season_id"] = season_id
                team_summary["source_url"] = page.url

                all_players.extend(players)
                all_team_rows.append(team_summary)

            except Exception as e:
                print("FAILED:", team, e)

            time.sleep(1.0)

        browser.close()

    PLAYER_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEAM_OUT.parent.mkdir(parents=True, exist_ok=True)

    players_df = pd.DataFrame(all_players)
    team_df = pd.DataFrame(all_team_rows)

    players_df.to_csv(PLAYER_OUT, index=False)
    team_df.to_csv(TEAM_OUT, index=False)

    print("Saved players:", PLAYER_OUT, players_df.shape)
    print("Saved team summary:", TEAM_OUT, team_df.shape)


if __name__ == "__main__":
    main()
