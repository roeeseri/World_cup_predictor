from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_PATH = ROOT / "models" / f"production_model_daily_{STAMP}.joblib"
CONFIG_PATH = ROOT / "models" / f"production_config_daily_{STAMP}.json"

LATEST_MODEL_PATH = ROOT / "models" / "production_model_latest.joblib"
LATEST_CONFIG_PATH = ROOT / "models" / "production_config_latest.json"


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full daily World Cup update pipeline.")
    parser.add_argument("matchday_csv", help="CSV file with completed World Cup matches")
    parser.add_argument("--skip-update", action="store_true", help="Skip appending matchday CSV")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip rebuilding model_dataset via notebook")
    parser.add_argument("--skip-train", action="store_true", help="Skip training new model")
    parser.add_argument("--skip-simulate", action="store_true", help="Skip tournament simulation")
    parser.add_argument("--skip-audit", action="store_true", help="Skip audit checks")
    args = parser.parse_args()

    matchday_csv = Path(args.matchday_csv)
    if not matchday_csv.is_absolute():
        matchday_csv = ROOT / matchday_csv

    if not matchday_csv.exists() and not args.skip_update:
        raise FileNotFoundError(f"Missing matchday CSV: {matchday_csv}")

    if not args.skip_update:
        run([
            sys.executable,
            "scripts/update_world_cup_data.py",
            str(matchday_csv),
        ])

    if not args.skip_dataset:
        run([
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "notebooks/05_build_model_dataset.ipynb",
            "--output",
            "05_build_model_dataset_executed.ipynb",
            "--output-dir",
            "notebooks",
        ])

    if not args.skip_train:
        run([
            sys.executable,
            "-m",
            "src.models.train_production",
            "--model-type",
            "ensemble",
            "--save-path",
            str(MODEL_PATH.relative_to(ROOT)),
            "--config-path",
            str(CONFIG_PATH.relative_to(ROOT)),
            "--no-evaluate",
        ])

        shutil.copy2(MODEL_PATH, LATEST_MODEL_PATH)
        shutil.copy2(CONFIG_PATH, LATEST_CONFIG_PATH)

        print("\nSaved daily model:")
        print(MODEL_PATH)
        print(CONFIG_PATH)
        print("Updated latest model:")
        print(LATEST_MODEL_PATH)
        print(LATEST_CONFIG_PATH)

    if not args.skip_simulate:
        run([sys.executable, "scripts/simulate_2026_world_cup.py"])

    if not args.skip_audit:
        run([sys.executable, "scripts/audit/audit_2026_simulation.py"])
        run([sys.executable, "scripts/audit/full_project_audit.py"])

    print("\nDAILY UPDATE PIPELINE FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
