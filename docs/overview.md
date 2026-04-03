# overview.md — Sign Language Interaction Environment (SLIE)

---

## 1. Problem Definition

Most AI-powered digital assistants today accept only **text or voice input**. This excludes millions of deaf or non-verbal users who communicate primarily through **sign language**.

SLIE solves this by creating a structured simulation environment where an AI agent must:
- Receive hand-sign landmark observations
- Interpret their meaning (intent)
- Execute appropriate actions
- Maintain multi-step conversational context

SLIE is **not** a raw computer vision project. It does not process images or video at runtime. Instead, it exposes deterministic hand landmark sequences and compact gesture embeddings so agents must solve perception without the environment leaking the true symbolic label.

---

## 2. Why It Matters (Accessibility Angle)

| Fact | Impact |
|------|--------|
| ~70 million deaf people worldwide use sign language | Massive underserved population |
| Existing AI assistants (Siri, Alexa, Google) require voice/text | Complete exclusion of non-verbal users |
| No standardized benchmark exists for sign-language AI agents | No way to measure progress |

SLIE provides:
- A **standardized evaluation environment** for sign-language-aware agents
- A **training ground** for LLM-based agents to learn gesture-to-intent mapping
- A **reproducible benchmark** usable by the research community

---

## 3. What Makes SLIE Different from Gesture Recognition Systems

| Gesture Recognition Systems | SLIE |
|-----------------------------|------|
| Goal: classify a gesture into a label | Goal: reason about intent, execute actions |
| Single-step classification | Multi-step interaction with memory |
| Input: raw images/video | Input: hand landmark sequences + compact embeddings |
| Output: gesture label | Output: intent + action + response |
| No environment loop | Full OpenEnv step/reset/state loop |
| No reward signal | Dense reward function across trajectory |

SLIE is an **agent reasoning environment**, not a classifier benchmark.

---

## 4. System Summary (End-to-End Flow)

```
[Pre-baked Gesture Prototype Dataset]
         │
         ▼
[Gesture Input Layer]         ← Builds landmark trajectories from static prototypes
         │
         ▼
[SLIE OpenEnv Environment]    ← Manages state, tasks, episodes
         │  (observation)
         ▼
[AI Agent (LLM-based)]        ← Receives observation, produces action
         │  (action)
         ▼
[SLIE Environment step()]     ← Evaluates action, computes reward
         │  (reward, done, next observation)
         ▼
[Grader]                      ← Scores episode on task completion
         │
         ▼
[Logs: START / STEP / END]    ← stdout logging for evaluation pipeline
```

---

## 5. Key Concepts

### Gesture
A pre-defined hand-sign class used internally by the environment.
- Example: `"HELLO"`, `"YES"`, `"OPEN"`, `"FOOD"`
- Not exposed directly to the agent at runtime
- Each gesture has a fixed 64-dim prototype embedding which is expanded into a deterministic landmark sequence

### Intent
The semantic meaning the agent assigns to a gesture or gesture sequence.
- Example: gesture `"HELLO"` → intent `"greeting"`
- Example: gesture sequence `["OPEN", "YOUTUBE"]` → intent `"open_app:youtube"`

### Action
What the agent does in response to the interpreted intent.
- Example: intent `"open_app:youtube"` → action response `"Opening YouTube"`

### Episode
One full task run: from `reset()` to `done=True`.
- Each episode has a fixed task (Task 1, 2, or 3)
- Each episode has a fixed gesture sequence (loaded from dataset)
- Max steps per episode: **10**

---

## 6. Scope Boundaries

**IN SCOPE:**
- Deterministic hand landmark sequence processing
- Intent mapping (lookup + LLM reasoning)
- Multi-step interaction with memory
- Reward computation
- Graded evaluation (3 tasks)
- OpenEnv API compliance
- Docker deployment
- HuggingFace Spaces deployment

**OUT OF SCOPE:**
- Raw image/video processing
- Real-time webcam input
- MediaPipe or any CV library at runtime
- Full sign language grammar
- Multilingual sign support
- Emotion detection
