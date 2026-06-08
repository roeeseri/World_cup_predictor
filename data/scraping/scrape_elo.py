from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path

year = "2026"
url = f"https://www.eloratings.net/{year}_results"
selector = f"#maintable_{year}_results"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # use False first for debugging
    page = browser.new_page()

    page.goto(url, wait_until="networkidle")
    page.wait_for_selector(selector, timeout=20000)

    table_text = page.locator(selector).inner_text()
    table_html = page.locator(selector).inner_html()

    browser.close()

# print(table_text[:3000])

# with open("elo_2025_results_text.txt", "w", encoding="utf-8") as f:
#     f.write(table_text)

# with open("elo_2025_results_table.html", "w", encoding="utf-8") as f:
#     f.write(table_html)

# Parse text file into DataFrame
lines = [line.strip() for line in table_text.split('\n') if line.strip()]
data_lines = lines[5:]  # Skip header lines

matches = []
i = 0
while i < len(data_lines):
    if i + 15 >= len(data_lines):
        break

    try:
        month_day = data_lines[i]
        year = data_lines[i + 1]
        team_a = data_lines[i + 2]
        team_b = data_lines[i + 3]
        goals_a = int(data_lines[i + 4])
        goals_b = int(data_lines[i + 5])
        competition = data_lines[i + 6]
        location = data_lines[i + 7].replace('in ', '')

        def parse_signed_int(s):
            s = s.replace('\u2212', '-').replace('−', '-').strip()
            return 0 if s == '-' else int(s)

        rating_change_a = parse_signed_int(data_lines[i + 8])
        rating_change_b = parse_signed_int(data_lines[i + 9])
        rating_a = int(data_lines[i + 10])
        rating_b = int(data_lines[i + 11])
        rank_change_a = parse_signed_int(data_lines[i + 12])
        rank_change_b = parse_signed_int(data_lines[i + 13])
        rank_a = int(data_lines[i + 14])
        rank_b = int(data_lines[i + 15])

        date_str = f"{month_day} {year}"
        date = pd.to_datetime(date_str, format='%B %d %Y')

        matches.append({
            'date': date,
            'team_a': team_a,
            'team_b': team_b,
            'goals_a': goals_a,
            'goals_b': goals_b,
            'competition': competition,
            'location': location,
            'rating_change_a': rating_change_a,
            'rating_change_b': rating_change_b,
            'rating_a': rating_a,
            'rating_b': rating_b,
            'rank_change_a': rank_change_a,
            'rank_change_b': rank_change_b,
            'rank_a': rank_a,
            'rank_b': rank_b,
        })

        i += 16
    except (ValueError, IndexError):
        break

df = pd.DataFrame(matches)

# Create data/raw directory and save files
raw_dir = Path('data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(raw_dir / f'elo_{year}_results.csv', index=False)
# df.to_parquet(raw_dir / f'elo_{year}_results.parquet')

print(f"Parsed {len(df)} matches and saved to data/raw/")