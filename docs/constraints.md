# constraints.md — Scope Boundaries and Constraints

---

## 1. What Is OUT OF SCOPE

The following are explicitly **not part of SLIE** and must NOT be implemented:

| Out of Scope Item | Reason |
|-------------------|--------|
| Real-time webcam or video input | Hardware dependency, CV complexity, not needed |
| MediaPipe or any CV library at runtime | Hardware/GPU dependency, out of resource budget |
| OpenCV processing | Same as above |
| Raw image or video file processing | Same as above |
| Full ASL/BSL grammar parsing | Too complex, not reproducible |
| Multilingual sign language support | Out of scope for v1 |
| Emotion detection from gestures | Out of scope |
| Real application integration (actual YouTube, Maps, etc.) | Simulation only |
| Fuzzy string matching in graders | Non-deterministic, exploitable |
| LLM-as-judge in graders | Non-deterministic, expensive |
| Multi-user or concurrent episode support | Single-threaded only |
| User authentication | Not needed for evaluation environment |
| Persistent storage (database) | All state is in-memory |
| GPU inference inside environment | 2vCPU/8GB constraint |
| Streaming responses | Standard request/response only |
| WebSocket support | Not needed |

---

## 2. Simplifications (What We Do Instead)

| Real-World Thing | SLIE Simplification | Why |
|-----------------|---------------------|-----|
| Hand landmark detection | Deterministic landmark trajectories generated from static gesture prototypes | Reproducibility, no hardware needed |
| Continuous gesture stream | Fixed-length landmark frame sequences per gesture | Simplicity, determinism |
| Ambiguous real signs | Controlled ambiguity via curated scenario design | Reliable grading |
| Real app execution | String response confirming simulated action | Sandbox safety |
| Continuous user input | Fixed gesture sequences per scenario | Reproducibility |
| Dynamic gesture vocabulary | Fixed 24-gesture vocabulary | Reliability |

---

## 3. Assumptions (Must Be True for System to Work)

| Assumption | Impact if Violated |
|------------|-------------------|
| `data/gestures.json` exists and is valid | `reset()` will fail with 500 error |
| `data/tasks.json` exists and has all 3 tasks with 5 scenarios each | `reset()` will fail |
| Each gesture in `gestures.json` has exactly 64 float values in `frame_features` | Observation schema validation will fail |
| `episode_seed % 5` always maps to a valid scenario | Must have exactly 5 scenarios per task |
| Environment runs as single process (no concurrent requests) | Race conditions in state if violated |
| Agent calls `reset()` before `step()` | `step()` returns 400 error otherwise |
| Agent calls `reset()` after `done=True` to start new episode | `step()` returns 400 error otherwise |
| `inference.py` is in root directory | Evaluation pipeline will not find it otherwise |
| Port 8000 is available | Docker `run -p 8000:8000` will fail otherwise |
| `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` env vars are set | `inference.py` will fail |

---

## 4. Known Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| Fixed gesture vocabulary (24 gestures) | Cannot add new gestures without editing `gestures.json` | Sufficient for evaluation |
| No multi-turn correction | Environment does not re-present a gesture if agent was wrong | Agent must recover on next step |
| Scenario coverage limited (5 per task) | Only 15 total scenarios | Sufficient for reproducible baseline |
| No real-time feedback on partial response quality | Response scoring is keyword-based only | Sufficient for grading purposes |
| Landmark sequences are synthetic, not measured from real signers | Less realistic than captured motion | Keeps the environment deterministic and lightweight |
| History limited to last 5 entries | Long sequences may lose early context | Max sequence is 8 gestures; sufficient |
| Single process — one episode at a time | Cannot evaluate multiple agents simultaneously | Run sequentially |

---

## 5. Resource Constraints

| Resource | Limit | Implementation Requirement |
|----------|-------|---------------------------|
| vCPU | 2 | FastAPI single worker, no parallel processing |
| Memory | 8GB | No large models loaded in environment; LLM is external |
| Inference time | <20 minutes total | MAX_STEPS=10 per task × 3 tasks = 30 steps max |
| Docker image size | <2GB recommended | Use `python:3.11-slim` base image |
| Network | Outbound to LLM API only | No other outbound calls from environment |

---

## 6. Environment vs Agent Boundaries

**Critical:** The environment and agent are **separate processes**.

| Concern | Environment (server) | Agent (inference.py) |
|---------|---------------------|---------------------|
| LLM calls | ❌ Never | ✅ Always |
| Grading | ✅ Always | ❌ Never |
| State management | ✅ Always | ❌ Never reads state directly |
| HTTP server | ✅ Runs FastAPI | ❌ Makes HTTP requests |
| stdout logging | ❌ No [START]/[STEP]/[END] | ✅ Required |
| `gestures.json` | ✅ Reads directly | ❌ No access needed |
| `tasks.json` | ✅ Reads directly | ❌ No access needed |

---

## 7. Data Integrity Rules

- Gesture labels are **case-sensitive** in `gestures.json` and `tasks.json` (use UPPERCASE always)
- Intent strings in grader comparisons are normalized to **lowercase** before comparison
- `frame_features` in `gestures.json` must always be exactly **64 floats** — no more, no less
- Scenario IDs must be integers **0 through 4** — no gaps
- All float scores must be **rounded to 4 decimal places** before storage
- `episode_seed` is an integer — floats or strings will cause 422 validation error
