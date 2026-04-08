---
title: SLIE
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# SLIE — Sign Language Interaction Environment

SLIE is a real-world OpenEnv environment for evaluating agents that interpret sign-language gesture streams and convert them into assistive actions. The environment models a practical accessibility workflow: a deaf or hard-of-hearing user communicates intent through gestures, and an agent must infer, track context, and respond correctly over multi-step interaction.

## Why This Environment

- Real-world utility: accessibility assistant behavior under ambiguous, sequential non-text input.
- Deterministic grading: repeatable benchmarking for model comparison.
- Progressive challenge: single-step classification -> contextual sequencing -> compound intent synthesis.

## OpenEnv API

SLIE exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

Core typed models are defined in [`slie/models.py`](/home/vagabond/Project/SLIE/slie/models.py).

## Observation Space

`SLIEObservation` includes:

- `detected_gesture: str | null`
- `gesture_embedding: list[float]` (64-dim)
- `hand_landmarks: list[HandFrame]` (21 points per hand)
- `context`:
  - `current_task: str`
  - `step_count: int`
  - `history: list[str]` (rolling window)

Observation generation is deterministic but realism-aware: for a fixed `episode_seed`, SLIE applies reproducible sensor jitter and sparse landmark dropout to better mimic webcam/pose-estimation noise.

## Action Space

`SLIEAction` includes:

- `intent: str` (max 100 chars)
- `confidence: float` (clamped to `[0.0, 1.0]`)
- `response: str` (max 500 chars)

## Tasks and Difficulty

- `task1` (easy): Command Recognition
- `task2` (medium): Multi-step Interaction
- `task3` (hard): Intent Inference (compound final intent)

Each task has 5 deterministic scenarios in [`data/tasks.json`](/home/vagabond/Project/SLIE/data/tasks.json), each with explicit grader objectives.

## Reward and Grading

- Per-step dense rewards in [`slie/reward.py`](/home/vagabond/Project/SLIE/slie/reward.py):
  - Intent match reward
  - Keyword-grounding reward
  - Final-step completion bonus
  - Loop/invalid-action penalties
- Episode final scores in [`slie/graders.py`](/home/vagabond/Project/SLIE/slie/graders.py), normalized to `[0.0, 1.0]`.
  - `task3` uses a weighted rubric with deterministic sub-scores:
    - compound intent correctness
    - keyword coverage
    - intermediate-step consistency
    - confidence calibration

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Docker

```bash
docker build -t slie .
docker run --rm -p 8000:8000 slie
```

## Baseline Inference

`inference.py` is the submission baseline runner and emits strict structured logs:

- `[START] task=<task> env=slie model=<model>`
- `[STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,...,rn>`

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (preferred) or `OPENAI_API_KEY`
- `ENV_URL` (optional, default `http://localhost:8000`)
- `EPISODES_PER_TASK` (optional, default `3`)
- `LOCAL_IMAGE_NAME` (optional; only for `from_docker_image()` style inference flows)

Note:
- The project always uses the OpenAI Python client.
- `API_BASE_URL` decides provider routing. With `https://router.huggingface.co/v1`, both Qwen and DeepSeek model IDs are valid in `MODEL_NAME`.

Run:

```bash
python inference.py
```

`inference.py` runs multiple seeded episodes per task and writes aggregate mean/std summary to `stderr` while keeping `stdout` strictly in `[START] / [STEP] / [END]` format.

## Baseline Scores (Reference)

Reference local run (`episode_seed=0`, `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`) reports one score per task in `[0.0, 1.0]` and is reproducible given fixed environment/task seed and deterministic client settings (`temperature=0.0`).

## Validation and Tests

```bash
python tests/spec_conformance.py
python tests/run_all_checks.py
pytest -q
```
