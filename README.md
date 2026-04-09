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

### MiniMax (recommended for `crl-report`)

Copy `.env.example` to `.env` and set your MiniMax API key (OpenAI-compatible endpoint):

```bash
cp .env.example .env
# Edit .env: MINIMAX_API_KEY, optional MINIMAX_MODEL (default MiniMax-M2)
```

See [MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api).

Optional CrewAI report (OpenAI or other providers supported by CrewAI):

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
