# Curriculum RL Lab

**Idea:** a *real* reinforcement-learning codebase (not a blank template): sparse-reward **grid navigation** trained with **PPO** (Stable-Baselines3 + PyTorch), plus an optional **CrewAI** “lab analyst” that turns `metrics.json` into a short narrative when `OPENAI_API_KEY` is set.

```mermaid
flowchart LR
    CLI{{"💻 crl-train · crl-report"}}
    ENV["🟦 env_bootstrap.py<br/>CurriculumGridWorldEnv"]
    TRAIN["🎯 training.py<br/>PPO loop"]
    METR["📊 metrics.py"]
    OUT[/"📂 runs/latest/<br/>model.zip · metrics.json"/]
    REP["📝 reporting_minimax<br/>+ reporting_crew"]
    LLM(("🤖 MiniMax / OpenAI"))
    MD[/"📄 report.md"/]

    CLI --> ENV --> TRAIN --> METR --> OUT
    OUT --> REP --> LLM
    REP --> MD

    classDef io fill:#0e1116,stroke:#2f81f7,stroke-width:1.5px,color:#e6edf3;
    classDef brain fill:#161b22,stroke:#d29922,stroke-width:1.5px,color:#e6edf3;
    classDef tool fill:#161b22,stroke:#3fb950,stroke-width:1.5px,color:#e6edf3;
    classDef out fill:#0e1116,stroke:#a371f7,stroke-width:1.5px,color:#e6edf3;
    class CLI brain;
    class LLM io;
    class ENV,TRAIN,METR,REP tool;
    class OUT,MD out;
```

## Table of contents

- [What you get](#what-you-get)
- [Setup](#setup)
- [Train](#train)
- [Report](#report)
- [Tests](#tests)
- [Project layout](#project-layout)
- [License](#license)

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

## Project layout

```
src/curriculum_rl_lab/
  cli.py                    # crl-train · crl-report entrypoints
  env_bootstrap.py          # CurriculumGridWorldEnv
  training.py               # PPO training loop
  metrics.py                # JSON metrics writer
  reporting_minimax.py      # MiniMax report path
  reporting_crew.py         # CrewAI report path
tests/                      # pytest suite
pyproject.toml              # [dev] + [crew] extras
```

## License

MIT
