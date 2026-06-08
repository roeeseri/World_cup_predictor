from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_2026_PATH = PROJECT_ROOT / "data/raw/elo_2026_results.csv"
UPDATES_MASTER_PATH = PROJECT_ROOT / "data/raw/world_cup_updates/all_world_cup_2026_updates.csv"
BACKUP_DIR = PROJECT_ROOT / "data/backups"

REQUIRED_COLS = [
    "date",
    "team_a",
    "team_b",
    "goals_a",
    "goals_b",
    "competition",
    "location",
    "rating_change_a",
    "rating_change_b",
    "rating_a",
    "rating_b",
    "rank_change_a",
    "rank_change_b",
    "rank_a",
    "rank_b",
]


def run_command(command: list[str]) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_new_matches(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    for col in ["goals_a", "goals_b", "rank_a", "rank_b"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def backup_file(path: Path) -> None:
    if not path.exists():
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved: {backup_path}")


def append_matches(new_matches_path: Path) -> pd.DataFrame:
    new_df = pd.read_csv(new_matches_path)
    new_df = validate_new_matches(new_df)

    if not RAW_2026_PATH.exists():
        raise FileNotFoundError(f"Missing base file: {RAW_2026_PATH}")

    old_df = pd.read_csv(RAW_2026_PATH)
    old_df["date"] = pd.to_datetime(old_df["date"], errors="coerce")

    backup_file(RAW_2026_PATH)

    combined = pd.concat([old_df, new_df], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["date", "team_a", "team_b", "competition"],
        keep="last",
    )

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(RAW_2026_PATH, index=False)

    UPDATES_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    if UPDATES_MASTER_PATH.exists():
        updates_old = pd.read_csv(UPDATES_MASTER_PATH)
        updates_old["date"] = pd.to_datetime(updates_old["date"], errors="coerce")
        updates = pd.concat([updates_old, new_df], ignore_index=True)
        updates = updates.drop_duplicates(
            subset=["date", "team_a", "team_b", "competition"],
            keep="last",
        )
    else:
        updates = new_df

    updates = updates.sort_values("date").reset_index(drop=True)
    updates.to_csv(UPDATES_MASTER_PATH, index=False)

    print(f"Added/updated {len(new_df)} matches")
    print(f"Updated raw file: {RAW_2026_PATH}")
    print(f"Updated master updates file: {UPDATES_MASTER_PATH}")

    return new_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new_matches_csv", help="CSV with new World Cup matches")
    parser.add_argument("--simulate", action="store_true", help="Run 2026 simulation after update")
    parser.add_argument("--no-simulate", action="store_true", help="Do not run simulation")
    args = parser.parse_args()

    new_matches_path = Path(args.new_matches_csv)
    if not new_matches_path.is_absolute():
        new_matches_path = PROJECT_ROOT / new_matches_path

    append_matches(new_matches_path)

    # This script only appends new completed matches.
    # Full rebuild/train/simulate is handled by scripts/run_daily_update.py.
    print("\nDone.")


if __name__ == "__main__":
    main()
