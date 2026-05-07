"""End-to-end pipeline runner.

Stages:
  1. download   2. validate   3. profile   4. split
  5. tune       6. train+eval (using best params, with live MLflow / W&B logging)
  7. compare    8. select winner
  9. register   10. report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import models as models_module
from .benchmark import benchmark
from .config import load_config, write_config_snapshot
from .dashboards import make_observer, wandb_session
from .data.download import download_dataset
from .data.load import load_movielens
from .data.profile import profile, write_profile
from .data.split import make_split
from .data.validate import validate
from .eval.runner import evaluate_model
from .features import build_feature_store
from .logging_setup import configure
from .paths import make_run_paths
from .registry.mlflow_registry import (
    log_finalize_run,
    register_winner,
    setup_tracking,
    start_model_run,
)
from .report.generator import render
from .seed import set_global_seed
from .tuning.tuner import tune_model, write_tuning_summary

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("movielense")
    p.add_argument("--config", default="config.yaml", type=Path)
    p.add_argument("--smoke", action="store_true",
                   help="fast mode: tiny tuning, fewer epochs (used in CI)")
    p.add_argument("--skip-tuning", action="store_true",
                   help="skip Optuna search and use config defaults")
    return p.parse_args(argv)


def apply_smoke_overrides(cfg_raw: dict) -> dict:
    cfg_raw["tuning"]["enabled"] = True
    cfg_raw["tuning"]["trials_per_model"] = 3
    cfg_raw["tuning"]["timeout_seconds_per_model"] = 60
    cfg_raw["models"]["bpr"]["epochs"] = 3
    cfg_raw["models"]["als"]["iterations"] = 5
    cfg_raw["models"]["lgbm"]["n_estimators"] = 60
    cfg_raw["models"]["lgbm"]["num_leaves"] = 31
    cfg_raw["models"]["sasrec"]["epochs"] = 5
    cfg_raw["models"]["sasrec"]["batch_size"] = 64
    # In smoke mode, skip slow tuning of lgbm and sasrec — defaults are good enough.
    cfg_raw["tuning"]["models_to_tune"] = [
        m for m in cfg_raw["tuning"]["models_to_tune"] if m not in ("lgbm", "sasrec")
    ]
    cfg_raw["benchmark"]["n_users_to_score"] = 50
    cfg_raw["benchmark"]["candidates_per_user"] = 500
    return cfg_raw


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg.raw = apply_smoke_overrides(cfg.raw)

    paths = make_run_paths(cfg.get("run_name"))
    configure(log_file=paths.log_file)
    log.info("run_id=%s artifacts=%s", paths.run_id, paths.root)

    set_global_seed(cfg.seed)
    write_config_snapshot(cfg, paths.config_snapshot)

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    data_dir = download_dataset(cfg["dataset"]["url"], raw_dir)
    ds = load_movielens(cfg["dataset"]["name"], data_dir)

    vr = validate(
        ds,
        min_user_interactions=cfg["dataset"]["min_user_interactions"],
        min_item_interactions=cfg["dataset"]["min_item_interactions"],
    )
    if not vr.ok:
        log.error("dataset validation failed: %s", vr.issues)
        return 1

    prof = profile(ds)
    write_profile(prof, paths.profile_json)

    splits = make_split(ds.ratings, cfg["split"])
    paths.split_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_parquet(paths.split_dir / "train.parquet")
    splits.val.to_parquet(paths.split_dir / "val.parquet")
    splits.test.to_parquet(paths.split_dir / "test.parquet")

    train_pos = splits.positives_only(splits.train)
    val_pos = splits.positives_only(splits.val)
    test_pos = splits.positives_only(splits.test)
    log.info(
        "positives -> train=%d val=%d test=%d",
        len(train_pos), len(val_pos), len(test_pos),
    )

    eval_cfg = cfg["eval"]
    sel_cfg = cfg["selection"]
    primary = sel_cfg["objective"]
    direction = sel_cfg["direction"]

    # Feature store needed by LightGBM (and ignored by other models).
    feature_store = build_feature_store(ds, train_positives=train_pos)

    tuning_results: dict[str, dict] = {}
    if cfg["tuning"]["enabled"] and not args.skip_tuning:
        for model_name in cfg["tuning"]["models_to_tune"]:
            log.info("tuning %s", model_name)
            tr = tune_model(
                model_name,
                train_positives=train_pos,
                val_positives=val_pos,
                train_for_seen=splits.train,
                num_users=ds.num_users,
                num_items=ds.num_items,
                seed=cfg.seed,
                n_trials=cfg["tuning"]["trials_per_model"],
                timeout=cfg["tuning"].get("timeout_seconds_per_model"),
                primary_metric=primary,
                direction=direction,
                eval_ks=eval_cfg["ks"],
                candidate_strategy=eval_cfg["candidate_strategy"],
                sampled_negatives=eval_cfg["sampled_negatives"],
                sampler=cfg["tuning"]["sampler"],
                pruner=cfg["tuning"]["pruner"],
                feature_store=feature_store,
            )
            write_tuning_summary(tr, paths.tuning_dir)
            tuning_results[model_name] = {
                "best_params": tr.best_params,
                "best_value": tr.best_value,
                "summary": tr.study_summary,
            }

    setup_tracking(cfg["registry"]["tracking_uri"], cfg["registry"]["experiment_name"])

    all_results: list[dict] = []
    benchmark_results: list[dict] = []
    feature_importance_payload: dict | None = None
    train_plus_val = pd.concat([splits.train, splits.val], ignore_index=True)

    with wandb_session(cfg.raw, paths.run_id) as wb_run:
        for model_name in cfg.models_enabled:
            params = dict(cfg["models"][model_name])
            if model_name in tuning_results:
                params.update(tuning_results[model_name]["best_params"])

            log.info("training %s with params=%s", model_name, params)
            with start_model_run(model_name, tags={"run_id": paths.run_id}) as ctx:
                observer = make_observer(cfg.raw, wb_run)
                model = models_module.build(
                    model_name,
                    params=params,
                    num_users=ds.num_users,
                    num_items=ds.num_items,
                    seed=cfg.seed,
                    feature_store=feature_store,
                )
                fit_t0 = time.perf_counter()
                model.fit(train_pos, observer=observer)
                fit_seconds = time.perf_counter() - fit_t0

                val_eval = evaluate_model(
                    model,
                    train=splits.train,
                    target_positives=val_pos,
                    num_items=ds.num_items,
                    ks=eval_cfg["ks"],
                    candidate_strategy=eval_cfg["candidate_strategy"],
                    sampled_negatives=eval_cfg["sampled_negatives"],
                    rng=np.random.default_rng(cfg.seed),
                )
                test_eval = evaluate_model(
                    model,
                    train=train_plus_val,
                    target_positives=test_pos,
                    num_items=ds.num_items,
                    ks=eval_cfg["ks"],
                    candidate_strategy=eval_cfg["candidate_strategy"],
                    sampled_negatives=eval_cfg["sampled_negatives"],
                    rng=np.random.default_rng(cfg.seed + 1),
                )
                log.info(
                    "%s: val %s=%.4f  test %s=%.4f",
                    model_name, primary, val_eval.metrics[primary],
                    primary, test_eval.metrics[primary],
                )

                log_finalize_run(
                    ctx,
                    model=model,
                    params=params,
                    val_metrics=val_eval.metrics,
                    test_metrics=test_eval.metrics,
                    artifacts={
                        "config": paths.config_snapshot,
                        "profile": paths.profile_json,
                    },
                )

                if wb_run is not None:
                    wb_run.log(
                        {f"{model_name}/val/{k}": v for k, v in val_eval.metrics.items()}
                        | {f"{model_name}/test/{k}": v for k, v in test_eval.metrics.items()}
                    )

                # Latency benchmark for batch production planning.
                if cfg["benchmark"].get("enabled"):
                    bench = benchmark(
                        model_name=model_name,
                        fit_seconds=fit_seconds,
                        model=model,
                        num_users=ds.num_users,
                        num_items=ds.num_items,
                        candidates_per_user=cfg["benchmark"]["candidates_per_user"],
                        n_users_to_score=cfg["benchmark"]["n_users_to_score"],
                        seed=cfg.seed,
                    ).to_dict()
                    benchmark_results.append(bench)
                    log.info(
                        "%s bench: fit=%.2fs  score p50=%.2fms  throughput=%.0f users/s  size=%.2fMB",
                        model_name, bench["fit_seconds"], bench["score_latency_p50_ms"],
                        bench["throughput_users_per_sec"], bench["serialized_mb"],
                    )

                # Feature importance — only the LGBM model exposes it.
                if model_name == "lgbm" and hasattr(model, "feature_importance"):
                    fi = model.feature_importance()
                    if fi is not None:
                        feature_importance_payload = {
                            "names": list(fi["names"]),
                            "gain": [float(x) for x in fi["gain"]],
                            "split": [float(x) for x in fi["split"]],
                        }

                all_results.append({
                    "name": model_name,
                    "params": params,
                    "val_metrics": val_eval.metrics,
                    "test_metrics": test_eval.metrics,
                    "is_baseline": model_name in sel_cfg["baselines"],
                    "mlflow_run_id": ctx.run_id,
                })

    # 7+8. compare and select
    non_baseline = [r for r in all_results if not r["is_baseline"]]
    pool = non_baseline if non_baseline else all_results
    if direction == "maximize":
        winner = max(pool, key=lambda r: r["test_metrics"][primary])
    else:
        winner = min(pool, key=lambda r: r["test_metrics"][primary])

    with paths.metrics_json.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with paths.selection_json.open("w") as f:
        json.dump(
            {
                "winner": winner["name"],
                "objective": primary,
                "direction": direction,
                "value": winner["test_metrics"][primary],
            },
            f,
            indent=2,
        )

    registered_version = register_winner(
        run_id=winner["mlflow_run_id"],
        registered_model_name=cfg["registry"]["registered_model_name"],
    )

    # Persist auxiliary artifacts.
    with (paths.root / "benchmark.json").open("w") as f:
        json.dump(benchmark_results, f, indent=2)
    if feature_importance_payload is not None:
        with (paths.root / "feature_importance.json").open("w") as f:
            json.dump(feature_importance_payload, f, indent=2)

    render(
        run_id=paths.run_id,
        config=cfg.raw,
        profile=prof,
        splits_summary=splits.summary(),
        model_results=all_results,
        tuning_results=tuning_results,
        winner=winner["name"],
        primary_metric=primary,
        tracking_uri=cfg["registry"]["tracking_uri"],
        registered_model_name=cfg["registry"]["registered_model_name"],
        registered_version=registered_version,
        out_html=paths.report_html,
        out_md=paths.report_md,
        benchmark_results=benchmark_results,
        feature_importance=feature_importance_payload,
    )

    latest = Path("artifacts/reports")
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "dashboard.html").write_text(paths.report_html.read_text())
    (latest / "research_report.md").write_text(paths.report_md.read_text())

    # GitHub Pages source — served from main:/docs.
    pages = Path("docs")
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "index.html").write_text(paths.report_html.read_text())

    log.info("done. winner=%s  %s=%.4f",
             winner["name"], primary, winner["test_metrics"][primary])
    log.info("artifacts: %s", paths.root)
    log.info("MLflow UI: run `make ui` then open http://localhost:%d",
             cfg["dashboards"]["mlflow"].get("ui_port", 5000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
