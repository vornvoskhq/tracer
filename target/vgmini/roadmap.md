# VGMini Refactor Roadmap

This roadmap defines a phased refactor to make VGMini fully configurable and extensible while preserving the current CLI workflow and focusing on EMA, MACD, and Heikin-Ashi indicators. Each sprint lists deliverables, acceptance criteria, and exact commands to run with expected outcomes for validation.

Key principles
- YAML-first configuration with deep-merge (global overridden by per-experiment)
- Simple, organized codebase (no external plugin system, no run registry)
- OHLCV from Postgres is the only data source
- Backtesting rules and execution are globally configured, but overridable in experiment where expressly allowed
- No GUI work in this phase; CLI remains the main interface

Planned folder structure
```
configs/
  global.yaml            # global defaults (db, universe, backtest, features, seeds)
  clusters.yaml          # optional: named symbol groups (e.g., big_tech)
  experiments/
    equal_weight_dual.yaml
    logistic_buy_only.yaml
    xgboost_buy_only.yaml
    ...
results/
  <experiment_name>/
    config_snapshot.yaml
    models/trained_model.pkl
    metrics.json
    feature_importance.json
    signals.parquet
    backtest.csv
    backtest_aggregate.json
    logs.txt
  latest_comparison.png
```

Architecture overview (targets for this refactor)
- FeatureBase: compute(df, params), metadata (type, bounds, lag), dependencies
- ModelBase: fit, predict_scores, feature_importance, save/load, supports_dual flag
- TargetStrategyBase: generate_targets (buy_only, dual_signal; later breakout, reversal)
- ExperimentBase: orchestrates data -> features -> targets -> model -> backtest -> artifacts
- Simple registries: features_registry, models_registry, targets_registry (str -> class)

Out of scope (for now)
- Third-party plugin system
- Web UI (Flask/Vite)
- MLflow/run registries
- GPU/cluster execution

---

Sprint 1 (Completed): Foundation and YAML migration

Status update (2025-09-10)
- YAML loader with deep-merge implemented (configs/global.yaml + configs/experiments/*.yaml)
- YAML experiment discovery in CLI (vgmini.py list)
- Standardized artifacts produced for equal_weight_dual:
  - config_snapshot.yaml, models/trained_model.pkl, signals.parquet, backtest.csv, backtest_aggregate.json, metrics.json, feature_importance.json, visualizations/*_performance.png, analysis/*
- Compare command operational; regenerates results/latest_comparison.png
- Lightweight base classes and registries scaffolded (src/architecture.py): FeatureBase, ModelBase, TargetStrategyBase, registries; Dual/BuyOnly target strategies provided
- CLI runs experiments by name via YAML (e.g., equal_weight_dual)
- Notes: Base classes/registries are non-invasive scaffolding; current pipeline remains source of truth
Goal: Establish config system, base classes, and migrate equal-weight experiment.

Deliverables
- YAML loader with deep-merge: configs/global.yaml + configs/experiments/*.yaml
- Discovery of experiments by filename under configs/experiments/
- Base classes and registries: FeatureBase, ModelBase, TargetStrategyBase, ExperimentBase
- Core features with dependencies auto-computed: EMA, MACD, Heikin-Ashi
- StandardExperiment pipeline using the new config, saving standardized artifacts
- Backward-compatible listing/runner that still recognizes legacy JSON in results/configs during transition

Acceptance criteria
- Running equal_weight_dual via YAML produces results/<name>/ with metrics.json, feature_importance.json, backtest.csv, models/
- CLI list discovers YAML experiments
- Compare works on completed experiments and saves results/latest_comparison.png

Commands to run
1) List available experiments
```
python vgmini.py list
```
Expected: A list that includes YAML-based experiments from configs/experiments/.

2) Run equal-weight baseline
```
python vgmini.py equal_weight_dual
```
Expected: results/equal_weight_dual/ created with artifacts:
- config_snapshot.yaml, metrics.json, feature_importance.json
- models/trained_model.pkl
- backtest.csv and backtest_aggregate.json
- signals.parquet

3) Compare all completed experiments
```
python vgmini.py compare
```
Expected: A console table of key metrics and a saved plot at results/latest_comparison.png.

Notes
- During migration, legacy JSON in results/configs remains discoverable but YAML is preferred.

---

Sprint 2 (Completed): Model wrappers, target strategies, and ensemble scaffold
Goal: Align models with ModelBase and support buy-only and dual target strategies.

Status update (2025-09-10)
- Target strategies scaffolded in src/architecture.py (DualSignalTargets, BuyOnlyTargets)
- Ensemble supported in pipeline via src/ml_models.EnsembleModel and YAML (base_models, weights)
- YAML experiments present: logistic_buy_only.yaml, xgboost_buy_only.yaml, ensemble_buy_only.yaml
- target_strategy is passed from YAML through vgmini into model_config and used by training/prediction

Key outcomes
- Target strategies are applied via registry at feature step (_apply_target_strategy)
- Robust handling for buy-only single-class cases; signals and backtests run without index errors
- SHAP fallback fixed; no blocking errors when explainers are unavailable
- Ensemble via YAML base_models/weights works; comparison updated

Acceptance criteria
- Buy-only and dual-signal experiments train and backtest end-to-end (validated)
- Ensemble experiment produces combined scores and artifacts (validated)

Commands to run
1) Run logistic buy-only
```
python vgmini.py logistic_buy_only
```
Expected: results/logistic_buy_only/ with metrics and artifacts; metrics.json should include buy-side model performance.

2) Run xgboost buy-only
```
python vgmini.py xgboost_buy_only
```
Expected: results/xgboost_buy_only/ with metrics and artifacts.

3) Compare select experiments
```
python vgmini.py compare logistic_buy_only xgboost_buy_only equal_weight_dual
```
Expected: Console comparison highlighting relative performance; plot updated.

---

Sprint 3 (Completed): Universe clusters, standardized results, and tests
Goal: Support named clusters and establish a stable results schema with basic tests.

Status update (2025-09-10)
- configs/clusters.yaml added; experiments can specify universe.cluster and override globals
- cluster-analysis.json supported as fallback source for clusters
- compare command writes results/summary/comparison_summary.(csv|html)
- Basic pytest coverage implemented (5 tests): YAML merge, cluster resolution, target strategies (dual/buy_only), and model robustness for single-class cases

Acceptance criteria
- Experiments referencing universe.cluster run successfully (validated with cluster_bigtech_equal_weight)
- Tests pass locally (5/5)

Commands to run
1) Define a cluster in configs/clusters.yaml and reference it in an experiment
```
# Example content in configs/clusters.yaml
clusters:
  big_tech: [MSFT, GOOGL, META, TSLA]

# Then run your experiment referencing
python vgmini.py equal_weight_dual
```
Expected: The experiment runs against the cluster-defined symbols if configured; artifacts as in Sprint 1.

2) Run tests
```
pytest -q
```
Expected: All tests pass; runtime under ~1 minute for basic unit tests.

---

Sprint 4 (Completed): Breakout/Reversal targets and calibration/tuning hooks
Goal: Add new target strategies and extensibility points for future calibration/tuning.

Status update (2025-09-10)
- Target strategies added and registered: buy, sell, buy_only, dual, dual_signal, breakout, reversal
- YAML target_params supported and passed to strategies
- Optional tuning supported for logistic (simple C grid); calibration hooks defined in config (default none)

Acceptance criteria
- Breakout/reversal target strategies usable via YAML with target_params (validated via registry path)
- Tuning toggled via YAML; defaults preserve current behavior

Commands to run
1) Run breakout and reversal experiments
```
python vgmini.py breakout
python vgmini.py reversal
```
Expected: Each produces a results/<name>/ folder with metrics and artifacts; compare can include them.

---

Sprint 5 (Completed): Backtesting refinements and comparison reporting
Goal: Improve standardized reporting and export comparison summaries.

Status update (2025-09-10)
- Dynamic threshold strategy 'dynamic_absolute' added; controlled by threshold_window, dynamic_k, and confidence_threshold
- Compare command writes CSV/HTML summaries to results/summary/
- Minor documentation improvements in code and YAML examples

Acceptance criteria
- Compare exports summary files (CSV/HTML) (validated)
- Dynamic thresholding available and configurable (validated)

Commands to run
1) Generate comparison summary
```
python vgmini.py compare
```
Expected: results/summary/ contains comparison CSV/HTML; console shows the same metrics; latest_comparison.png updated.

---

Configuration semantics
- Global config is authoritative for backtesting, data source, and shared parameters; experiments may override specific fields for flexibility (e.g., symbols/cluster, features.enabled, model type/params, target.strategy).
- Deep-merge: unspecified fields inherit from global; lists in experiments replace global lists unless marked as additive (future option).

Success criteria for the refactor
- Users define experiments entirely through YAML files without touching code
- Adding a new feature/model/target is a matter of subclassing and registering, with no changes to the rest of the system
- Reproducible runs using a single global seed in configs/global.yaml
- Clear, comparable results for all experiments without a hard-coded baseline

Risks and mitigations
- Migration complexity: keep JSON fallback discoverable during transition; provide YAML templates and examples
- Performance regressions: maintain vectorized feature computations; reuse existing backtesting engine
- Scope creep: keep to EMA/MACD/HA and core models; defer advanced tuners/calibration to hooks only

Release plan
- Tag each sprint completion with a git tag (e.g., v1.1-sprint1)
- Publish YAML templates and examples with each sprint

Contact and collaboration
- Use pytest for tests
- No external plugin system; keep core small and maintainable
- PR reviews focus on config UX, readability, and correctness
