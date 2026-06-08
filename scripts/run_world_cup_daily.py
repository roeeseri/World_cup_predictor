
from __future__ import annotations

import argparse

import subprocess

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> None:

    print("\n" + "=" * 80)

    print("RUNNING:", " ".join(cmd))

    print("=" * 80)

    subprocess.run(cmd, cwd=ROOT, check=True)

def main() -> None:

    parser = argparse.ArgumentParser(description="Run daily World Cup update without retraining.")

    parser.add_argument("matchday_csv", help="CSV file with completed matches")

    parser.add_argument("--skip-update", action="store_true")

    parser.add_argument("--skip-rebuild-features", action="store_true")

    parser.add_argument("--skip-simulate", action="store_true")

    parser.add_argument("--skip-audit", action="store_true")

    args = parser.parse_args()

    if not args.skip_update:

        run([sys.executable, "scripts/update_world_cup_data.py", args.matchday_csv])

    if not args.skip_rebuild_features:

        run([sys.executable, "scripts/rebuild_2026_features.py"])

    if not args.skip_simulate:

        run([sys.executable, "scripts/simulate_2026_world_cup.py"])

    if not args.skip_audit:

        run([sys.executable, "scripts/audit/audit_2026_simulation.py"])

    print("\nDaily World Cup update completed successfully.")

if __name__ == "__main__":

    main()

