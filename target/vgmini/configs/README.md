# VGMini YAML Configuration (Sprint 1)

Structure:
- configs/global.yaml: global defaults
- configs/experiments/<name>.yaml: per-experiment overrides

Deep-merge semantics: experiment overrides global. Lists replace by default.

Run examples:
- List: `python vgmini.py list`
- Run baseline: `python vgmini.py equal_weight_dual`
- Compare completed: `python vgmini.py compare`
