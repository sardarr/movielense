"""Generate HTML dashboard + markdown research report from a run's artifacts."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

log = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(default_for_string=False, default=False),
    )


def render(
    *,
    run_id: str,
    config: dict,
    profile: dict,
    splits_summary: dict,
    model_results: list[dict],
    tuning_results: dict[str, dict],
    winner: str,
    primary_metric: str,
    tracking_uri: str,
    registered_model_name: str,
    registered_version: str | None,
    out_html: Path,
    out_md: Path,
) -> tuple[Path, Path]:
    env = _env()

    metric_columns: list[str] = []
    if model_results:
        first = model_results[0]["test_metrics"]
        # stable ordering: sort by metric family then by k
        metric_columns = sorted(first.keys(), key=_metric_sort_key)

    model_rows = []
    for r in model_results:
        model_rows.append({
            "name": r["name"],
            "metrics": r["test_metrics"],
            "is_baseline": r.get("is_baseline", False),
            "is_winner": r["name"] == winner,
        })

    metric_bars = []
    for col in metric_columns:
        metric_bars.append({
            "x": [r["name"] for r in model_rows],
            "y": [r["metrics"].get(col, 0.0) for r in model_rows],
            "name": col,
            "type": "bar",
        })

    tuning_curves = []
    for model_name, summary in (tuning_results or {}).items():
        trials = summary.get("summary", {}).get("trials", [])
        x = [t["number"] for t in trials if t["value"] is not None]
        y = [t["value"] for t in trials if t["value"] is not None]
        if x:
            running_best: list[float] = []
            best = -float("inf")
            for v in y:
                best = max(best, v)
                running_best.append(best)
            tuning_curves.append({
                "x": x,
                "y": running_best,
                "name": model_name,
                "mode": "lines+markers",
                "type": "scatter",
            })

    rating_hist = profile.get("rating_hist", {})
    user_dist = profile.get("user_interactions", {})
    template = env.get_template("dashboard.html.j2")
    html = template.render(
        run_id=run_id,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
        seed=config.get("seed"),
        primary_metric=primary_metric,
        winner=winner,
        profile=profile,
        splits=splits_summary,
        model_rows=model_rows,
        metric_columns=metric_columns,
        metric_bars=metric_bars,
        tuning_curves=tuning_curves,
        best_params_pretty=json.dumps(
            {k: v.get("best_params", {}) for k, v in (tuning_results or {}).items()},
            indent=2,
        ),
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        git_commit=_git_commit(),
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        registered_version=registered_version,
        rating_hist_x=list(rating_hist.keys()),
        rating_hist_y=list(rating_hist.values()),
        user_deg_x=[user_dist.get("min", 0), user_dist.get("p25", 0),
                    user_dist.get("median", 0), user_dist.get("p75", 0),
                    user_dist.get("max", 0)],  # crude — see note in README
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)

    md = _render_markdown(
        run_id=run_id,
        config=config,
        profile=profile,
        splits=splits_summary,
        model_rows=model_rows,
        metric_columns=metric_columns,
        winner=winner,
        primary_metric=primary_metric,
        tuning_results=tuning_results or {},
        registered_model_name=registered_model_name,
        registered_version=registered_version,
    )
    out_md.write_text(md)

    log.info("report written: %s, %s", out_html, out_md)
    return out_html, out_md


def _metric_sort_key(name: str) -> tuple[int, int]:
    family_order = {"hr": 0, "precision": 1, "recall": 2, "map": 3, "mrr": 4, "ndcg": 5}
    if "@" in name:
        fam, k = name.split("@", 1)
        try:
            return (family_order.get(fam, 99), int(k))
        except ValueError:
            return (family_order.get(fam, 99), 0)
    return (99, 0)


def _render_markdown(
    *,
    run_id: str,
    config: dict,
    profile: dict,
    splits: dict,
    model_rows: list[dict],
    metric_columns: list[str],
    winner: str,
    primary_metric: str,
    tuning_results: dict[str, Any],
    registered_model_name: str,
    registered_version: str | None,
) -> str:
    lines = []
    lines.append("# MovieLense Research Report\n")
    lines.append(f"Run id: `{run_id}`  ")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}  ")
    lines.append(f"Primary metric: **{primary_metric}**  ")
    lines.append(f"Winner: **{winner}**\n")

    lines.append("## Dataset")
    lines.append(
        f"- Users: {profile.get('num_users')}\n"
        f"- Items: {profile.get('num_items')}\n"
        f"- Interactions: {profile.get('num_interactions')}\n"
        f"- Sparsity: {profile.get('sparsity'):.4f}\n"
    )

    lines.append("## Splits")
    lines.append(
        f"- Train: {splits['train']}\n- Val: {splits['val']}\n- Test: {splits['test']}\n"
    )

    lines.append("## Model comparison (test set)\n")
    header = "| Model | " + " | ".join(metric_columns) + " |"
    sep = "|" + "---|" * (1 + len(metric_columns))
    lines.append(header)
    lines.append(sep)
    for r in model_rows:
        cells = [f"{r['metrics'].get(c, 0.0):.4f}" for c in metric_columns]
        marker = " (winner)" if r["is_winner"] else ""
        lines.append(f"| {r['name']}{marker} | " + " | ".join(cells) + " |")
    lines.append("")

    if tuning_results:
        lines.append("## Best hyperparameters\n")
        for name, summary in tuning_results.items():
            lines.append(f"- **{name}**: `{summary.get('best_params')}` ({primary_metric}={summary.get('best_value'):.4f})")
        lines.append("")

    lines.append("## Registry")
    lines.append(
        f"- Registered model: `{registered_model_name}`"
        + (f" (v{registered_version})" if registered_version else " (registration skipped)")
    )

    lines.append("\n## Reproducibility\n")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(config, sort_keys=False).rstrip())
    lines.append("```")
    return "\n".join(lines) + "\n"
