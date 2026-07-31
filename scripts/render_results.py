"""Render the Results table in README.md from reports/metrics.json.

Run after training so the README never carries hand-typed numbers:

    rotten-review train-sentiment && rotten-review train-score \\
        && rotten-review train-anomaly
    python scripts/render_results.py

The script rewrites only the block between the two marker comments in README.md
and refuses to invent anything: metrics absent from metrics.json are simply not
listed.

Every file is read and written as UTF-8 explicitly. Python falls back to the
locale encoding otherwise, which is cp1252 on a default Windows install — and
this README contains em dashes, superscripts and plus-minus signs, so the
implicit path raises UnicodeDecodeError there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"
README_PATH = REPO_ROOT / "README.md"

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_table(metrics: dict) -> str:
    lines = [
        "| Model | Metric | Value | Evaluated on |",
        "|---|---|---|---|",
    ]

    sentiment = metrics.get("sentiment_en")
    if sentiment:
        lines.append(
            f"| Model 1 — sentiment | accuracy | {_fmt(sentiment.get('accuracy'))} "
            f"| {sentiment.get('n_eval', '—')} holdout reviews |"
        )

    score = metrics.get("score_regressor")
    if score:
        lines.append(
            f"| Model 2 — score from text | MAE | {_fmt(score.get('mae'))} "
            f"| {score.get('n_eval', '—')} holdout reviews |"
        )
        lines.append(
            f"| Model 2 — score from text | R² | {_fmt(score.get('r2'))} "
            f"| {score.get('n_eval', '—')} holdout reviews |"
        )
        lines.append(
            f"| Model 2 — score from text | within ±1 point "
            f"| {_fmt(score.get('within_1_point'))} "
            f"| {score.get('n_eval', '—')} holdout reviews |"
        )

    anomaly = metrics.get("anomaly")
    if anomaly:
        lines.append(
            f"| Model 3 — anomaly detection | reviews flagged "
            f"| {anomaly.get('flagged', '—')} ({_fmt(anomaly.get('flagged_pct'), 1)}%) "
            f"| {anomaly.get('n_reviews', '—')} reviews, "
            f"{anomaly.get('n_features', '—')} features |"
        )

    if len(lines) == 2:
        return "_No metrics recorded yet — run `make train`._"

    lines.append("")
    lines.append(
        "Model 3's flag rate follows directly from the `contamination` parameter and is not "
        "a measurement — see [AUDIT.md](AUDIT.md) and the notebook's limitations section."
    )
    return "\n".join(lines)


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"{METRICS_PATH} not found — run the training commands first.", file=sys.stderr)
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    readme = README_PATH.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        print(f"Markers {START} / {END} not found in README.md", file=sys.stderr)
        return 1

    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    README_PATH.write_text(
        f"{head}{START}\n\n{build_table(metrics)}\n\n{END}{tail}", encoding="utf-8"
    )
    print("README results table updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
