# SLIE — Sign Language Interaction Environment

SLIE is a deterministic OpenEnv-style environment for evaluating AI agents that interpret hand-sign observations without being given the true gesture label.

## Features

- FastAPI environment with `/reset`, `/step`, `/state`, `/health`
- Observation contract built around hand landmark sequences plus a compact gesture embedding
- Three tasks:
  - `task1`: Command Recognition
  - `task2`: Multi-step Interaction
  - `task3`: Intent Inference
- Dense per-step rewards + deterministic end-of-episode grading
- Baseline external agent in `inference.py` that decodes the perceptual observation before reasoning

## Project Structure

- `slie/`: core package (`models`, `state`, `env`, `reward`, `graders`, etc.)
- `data/`: `gestures.json` and `tasks.json`
- `tests/`: unit + integration tests
- `docs/`: specification docs copied from `AGENTS/`

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Health check:

```bash
curl http://localhost:8000/health
```

## API Quickstart

Reset:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1","episode_seed":0}'
```

Step:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"intent":"greeting","confidence":0.95,"response":"Hello, how can I help?"}'
```

Reset responses now include:
- `gesture_embedding`: fixed-width 64-dim perceptual embedding
- `hand_landmarks`: deterministic sequence of hand landmark frames
- `context.history`: prior agent intents only, without leaked gesture labels or correctness flags

## Baseline Agent

`inference.py` runs `task1`, `task2`, and `task3` sequentially against a running environment and logs:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

Required env vars (see `.env.example`):

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_URL` (optional, defaults to `http://localhost:8000`)

## Docker

```bash
docker build -t slie .
docker run -p 8000:8000 slie
```

## Test

```bash
pytest -q
```

If `pytest` is not installed in your active environment, run:

```bash
python tests/smoke_checks.py
```
