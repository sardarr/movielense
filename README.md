# MovieLense

Self-running recommendation research pipeline on MovieLens.

One command runs the full experiment: download data, validate, profile, split, train baselines + collaborative filtering + a torch BPR model, tune hyperparameters with Optuna, evaluate with ranking metrics, register the winner in MLflow, and emit an HTML dashboard plus a markdown research report.

## Quick start

```bash
make sync     # install deps (uses uv, pins Python 3.12)
make run      # run the full pipeline (default: ml-100k)
make ui       # launch MLflow UI at http://localhost:5000
make test     # run the test suite
```

Outputs land in `artifacts/` and the MLflow registry at `mlruns.db`.

## Larger dataset (ML-1M)

For 1M ratings × 6K users × 3.7K items:

```bash
uv run python -m movielense.run --config config.ml-1m.yaml
```

Same models, same metrics, longer training. Tuning is off by default in this config because each trial takes minutes on the full set; flip `tuning.enabled` to true if you have a budget.

## Dashboards

Three layers of visibility:

| Layer            | What it shows                                   | How to see it                              |
|------------------|--------------------------------------------------|--------------------------------------------|
| **MLflow UI**    | Live training curves, all past runs, registry   | `make ui` → http://localhost:5000          |
| **GitHub Actions** | Per-stage CI/pipeline status, logs            | Repo "Actions" tab on GitHub               |
| **GitHub Pages** | Final HTML dashboard published per pipeline run | `https://<user>.github.io/<repo>/`         |
| **W&B (opt-in)** | Hosted live dashboard, per-epoch streaming      | `make sync-wandb`, set `WANDB_API_KEY`, flip `dashboards.wandb.enabled: true` |

Per-epoch metrics (BPR loss, ALS reconstruction error) stream live to MLflow during training, so `make ui` shows curves updating in real time.

## Pipeline stages

1. **Download** MovieLens-100K (cached under `data/raw/`).
2. **Validate** schema and integrity.
3. **Profile** dataset (sparsity, distributions).
4. **Split** train / val / test (temporal per-user by default).
5. **Train** Random, Popularity, Item-kNN, ALS, BPR-torch.
6. **Tune** hyperparameters with Optuna (per model).
7. **Evaluate** with HR@K, NDCG@K, Recall@K, Precision@K, MAP@K, MRR.
8. **Compare** all models including baselines.
9. **Select** winner by configurable objective (default: NDCG@10).
10. **Register** the winning model in MLflow.
11. **Generate** an HTML dashboard and markdown research report.

## Configuration

Everything is in [`config.yaml`](./config.yaml). Change dataset, models, search trials, eval cutoffs, selection objective, registry URI — all there.

## Layout

```
src/movielense/
  data/        download + load + validate + profile + split
  models/      base, baselines, item-kNN, ALS, BPR (torch)
  eval/        ranking metrics
  tuning/      Optuna search
  registry/    MLflow tracking + model registry
  report/      HTML + markdown report generation
  run.py       orchestrator
config.yaml
tests/
```

## Reproducibility

Every run captures: random seeds, dataset hash, full config snapshot, model params, code commit (when in git). Re-running with the same config produces the same results.

## CI

GitHub Actions runs lint + tests on every push, and a smoke run of the pipeline on a smaller config.
