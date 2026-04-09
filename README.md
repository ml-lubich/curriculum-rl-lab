# Curriculum RL Lab

**Idea:** a *real* reinforcement-learning codebase (not a blank template): sparse-reward **grid navigation** trained with **PPO** (Stable-Baselines3 + PyTorch), plus an optional **CrewAI** “lab analyst” that turns `metrics.json` into a short narrative when `OPENAI_API_KEY` is set.

## What you get

- Custom **Gymnasium** environment `CurriculumGridWorldEnv` (sparse goal reward, step penalty).
- **Training** script that saves `model.zip` and `metrics.json`.
- **Reporting**: rule-based markdown by default; CrewAI layer if you install extras and provide an API key.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional CrewAI report:

```bash
pip install -e ".[crew]"
export OPENAI_API_KEY=...
```

## Train

```bash
crl-train --out runs/latest --timesteps 80000 --seed 0
```

## Report

```bash
crl-report --metrics runs/latest/metrics.json --out runs/latest/report.md
```

## Tests

```bash
pytest
```

## License

MIT
