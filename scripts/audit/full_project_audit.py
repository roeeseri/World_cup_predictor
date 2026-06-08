from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

REPORT_PATH = REPORTS / f"full_project_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"


COMMANDS = [
    ["git", "status"],
    ["git", "log", "--oneline", "-8"],
    [sys.executable, "-m", "py_compile",
     "src/tournament/match_simulation.py",
     "src/tournament/simulate_world_cup.py",
     "scripts/simulate_2026_world_cup.py",
     "scripts/update_world_cup_data.py"],
    [sys.executable, "scripts/simulate_2026_world_cup.py"],
    [sys.executable, "scripts/audit/audit_2026_simulation.py"],
]


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        return (
            f"$ {' '.join(cmd)}\n\n"
            f"RETURN CODE: {out.returncode}\n\n"
            f"STDOUT:\n{out.stdout}\n\n"
            f"STDERR:\n{out.stderr}\n"
        )
    except Exception as e:
        return f"$ {' '.join(cmd)}\n\nFAILED: {type(e).__name__}: {e}\n"


def file_info(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        return f"- `{path}`: MISSING"
    return f"- `{path}`: {p.stat().st_size:,} bytes, modified {datetime.fromtimestamp(p.stat().st_mtime)}"


def csv_summary(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        return f"### {path}\nMISSING\n"

    try:
        df = pd.read_csv(p)
        text = [
            f"### {path}",
            f"- shape: {df.shape}",
            f"- columns: {list(df.columns)}",
            "",
            "head:",
            "```",
            df.head().to_string(),
            "```",
            "",
            "missing values top:",
            "```",
            df.isna().sum().sort_values(ascending=False).head(20).to_string(),
            "```",
        ]

        if {"team_a", "team_b"}.issubset(df.columns):
            teams = sorted(set(df["team_a"].dropna()) | set(df["team_b"].dropna()))
            text.append(f"- unique teams: {len(teams)}")

        if "pred_score" in df.columns:
            text += ["", "score distribution:", "```", df["pred_score"].value_counts().to_string(), "```"]

        return "\n".join(text)

    except Exception as e:
        return f"### {path}\nFAILED TO READ: {type(e).__name__}: {e}\n"


def model_config_summary() -> str:
    parts = []
    for path in sorted((ROOT / "models").glob("production_config*.json")):
        try:
            data = json.loads(path.read_text())
            parts.append(
                f"### {path.relative_to(ROOT)}\n"
                f"- model_type: {data.get('model_type')}\n"
                f"- features: {len(data.get('feature_cols', []))}\n"
                f"- target_cols: {data.get('target_cols')}\n"
                f"- use_weights: {data.get('use_weights')}\n"
                f"- trained_on_date: {data.get('trained_on_date')}\n"
            )
        except Exception as e:
            parts.append(f"### {path.relative_to(ROOT)}\nFAILED: {e}\n")
    return "\n".join(parts)


def main() -> None:
    sections = []

    sections.append("# Full Project Audit")
    sections.append(f"Generated: {datetime.now()}")
    sections.append("")

    sections.append("## Critical files")
    for path in [
        "data/processed/model_dataset.csv",
        "data/processed/world_cup_2026_group_stage_features.csv",
        "data/raw/elo_2026_results.csv",
        "models/production_model_v3.joblib",
        "models/production_config_v3.json",
        "scripts/update_world_cup_data.py",
        "src/tournament/match_simulation.py",
        "src/tournament/simulate_world_cup.py",
        "src/app/streamlit_app.py",
    ]:
        sections.append(file_info(path))

    sections.append("\n## Model configs")
    sections.append(model_config_summary())

    sections.append("\n## CSV summaries")
    for path in [
        "data/processed/model_dataset.csv",
        "data/processed/world_cup_2026_group_stage_features.csv",
        "outputs/evaluation/world_cup_2026_simulation/group_predictions.csv",
        "outputs/evaluation/world_cup_2026_simulation/group_standings.csv",
        "outputs/evaluation/world_cup_2026_simulation/knockout_results.csv",
    ]:
        sections.append(csv_summary(path))

    sections.append("\n## Command checks")
    for cmd in COMMANDS:
        sections.append("```text")
        sections.append(run_cmd(cmd))
        sections.append("```")

    REPORT_PATH.write_text("\n\n".join(sections))
    print(f"Audit report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
