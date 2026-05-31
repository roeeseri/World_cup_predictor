# Notebook: 06_scrape_transfermarkt_market_values.ipynb

## Purpose
Scrapes squad market values from Transfermarkt for international teams across years 2004-2026.

## What's Done
- Uses Playwright (browser automation) + BeautifulSoup to scrape Transfermarkt pages
- Collects squad market values by national team and year
- Outputs raw scraped data, consumed by `02_feature_engineering.ipynb` for cleaning

## Dependencies
- Playwright (requires browser binaries installed)
- BeautifulSoup4
- Active internet connection + Transfermarkt accessibility

## Status
Complete. Data already scraped and cleaned. Do not re-run unless updating to current values.
Re-running takes several hours and depends on Transfermarkt's scraping policies.

## Output
Raw scraped file used as input to notebook 02. Final cleaned output is `data/processed/transfermarkt_market_values_clean.csv`.
