from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

tm = pd.read_csv("data/processed/transfermarkt_market_values_clean.csv")

row = tm[
    (tm["season_id"] == 2026) &
    (tm["team_name_tm"] == "Argentina")
].dropna(subset=["url"]).iloc[0]

url = row["url"]

out_dir = Path("outputs/debug")
out_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    )

    print("Opening:", url)
    page.goto(url, wait_until="networkidle", timeout=60000)

    print("Loaded:", page.url)
    print("Title:", page.title())

    links = page.locator("a")
    detailed = []

    for i in range(min(links.count(), 300)):
        text = links.nth(i).inner_text().strip()
        href = links.nth(i).get_attribute("href")

        if "squad" in text.lower() or "squad" in str(href).lower():
            detailed.append((text, href))

    print("\nSquad-related links:")
    for item in detailed[:30]:
        print(item)

    html_path = out_dir / "argentina_transfermarkt_debug.html"
    html_path.write_text(page.content(), encoding="utf-8")

    print("\nSaved HTML:", html_path)
    print("table.items count:", page.locator("table.items").count())
    print("tbody rows:", page.locator("table.items tbody tr").count())
    print("td.posrela count:", page.locator("td.posrela").count())

    browser.close()
