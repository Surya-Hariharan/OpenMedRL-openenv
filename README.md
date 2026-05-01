# TriageRL
**Deterministic Medical Triage RL Environment with Reward-Driven Episode Simulation**  
*OpenEnv-style benchmark for ESI classification, clarifying questions, and safety-aware scoring*

## Submission-ready cleanup

- Removed dev/test artifacts and egg-info metadata for a leaner submission.
- Local virtual environments (e.g. `.venv/`) are excluded via `.gitignore` and should not be included in the submitted archive.
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-black)
![Reward](https://img.shields.io/badge/Reward-Deterministic-success)
![ESI](https://img.shields.io/badge/ESI-1--5-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-success)

---

## Mission Statement

**TriageRL** is a production-oriented medical triage environment for reinforcement learning agents. It simulates emergency-department decision making around the Emergency Severity Index (ESI 1-5), requiring the agent to either classify the case or ask a high-yield clarifying question when information is missing.

The system is built to be deterministic, reward-stable, and easy to audit. Episode generation, environment replay, and reward grading are all designed so the same prompt and completion produce the same score across runs.

---

## Platform Overview

TriageRL focuses on four core problems in triage training:

- **Deterministic reward assignment**: identical inputs always produce identical outputs.
- **Episode-level simulation**: prompts can represent both initial and mid-episode states.
- **Safety-aware grading**: undertriage is penalized more strongly than conservative reasoning.
- **Exploit resistance**: clarify actions are not profitable, and malformed outputs are rejected.

**Key properties:**
- Reward ordering is stable: Good > Stuffed > Clarify > Bad.
- No silent success path for malformed completions.
- Reward function is deterministic across repeated runs.
- Training data can be refreshed per epoch for on-policy style updates.

---

## Core Features

### Deterministic Reward Pipeline
- Prompt text includes embedded task and sample identifiers.
- `triagerl.training.reward_fn` reconstructs the episode deterministically.
- `compute_final_score()` is used for valid classify completions.
- Clarify completions receive a small negative reward to prevent exploit loops.

### Clinical Episode Simulation
- Structured triage scenarios cover cardiovascular, neurological, infectious, respiratory, and abdominal cases.
- Environment state includes partial observability and replayable clarify steps.
- Max-step termination is handled explicitly with timeout penalties.

### Reward Shaping and Safety
- Undertriage and weak reasoning are penalized.
- Keyword stuffing is discouraged through path-quality heuristics.
- Invalid JSON, empty outputs, and schema failures are rejected with explicit penalties.

### Training-Oriented Dataset Generation
- `TriageEpisodeDataset.refresh(epoch)` regenerates prompts each epoch.
- Mid-episode prompts support multi-step policy learning.
- Dataset generation is deterministic for a given seed and epoch.

### Lightweight API Surface
- FastAPI server exposes the application entrypoint.
- `triagerl.api.server:app` is the supported application object.
- The package is importable without the removed debug and evaluation scripts.

---

## System Architecture

TriageRL uses a compact, audit-friendly pipeline:

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt + Task Context                      │
│  Embedded task id, sample seed, and step context                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Reward Function / Replay Layer                │
│  Parse JSON → validate action → replay clarify steps           │
│  compute_final_score() for classify completions                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MedicalTriageEnv                          │
│  Episode state, vital drift, observability, timeout handling    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Task Corpus + Reward Modules                  │
│  Grader, shaping, path quality, safety, and metrics            │
└─────────────────────────────────────────────────────────────────┘
```

**Flow summary:**
1. A task is selected and embedded in the prompt.
2. The model produces either a classify or clarify action.
3. The environment or reward function replays episode context if needed.
4. `compute_final_score()` returns the final reward for valid classification.
5. Diagnostics capture penalties for malformed or adversarial outputs.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Python Runtime** | Python 3.11+ | Core package and training logic |
| **API Server** | FastAPI | Environment and session API |
| **Environment** | Custom triage simulator | Episode state and action handling |
| **Rewarding** | Deterministic grading pipeline | Stable scalar feedback |
| **Training** | TRL + Unsloth (optional) | GRPO-style fine-tuning |
| **Validation** | Pydantic | Action and schema validation |
| **Task Corpus** | YAML task definitions | Scenario catalog and metadata |
| **Logging** | Standard logging | Diagnostics and audit trails |

---

## Project Structure

```text
TriageRL/
├── triagerl/
│   ├── api/               # FastAPI app and endpoints
│   ├── core/              # Shared models and constants
│   ├── env/               # Episode simulation and observability
│   ├── eval/              # Evaluation helpers and smoke checks
│   ├── logs/              # Logging helpers
│   ├── reward/            # Reward grading and shaping
│   ├── tasks/             # Task loading and schema
│   └── training/          # Dataset, prompt builder, reward fn, train loop
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── openenv.yaml
```

**Key files:**
- [`triagerl/api/server.py`](triagerl/api/server.py) - FastAPI application and `main()` entrypoint.
- [`triagerl/env/triage_env.py`](triagerl/env/triage_env.py) - environment control flow and termination logic.
- [`triagerl/reward/grader.py`](triagerl/reward/grader.py) - final score aggregation.
- [`triagerl/training/reward_fn.py`](triagerl/training/reward_fn.py) - deterministic reward computation.
- [`triagerl/training/dataset.py`](triagerl/training/dataset.py) - prompt and epoch refresh logic.
- [`triagerl/training/train.py`](triagerl/training/train.py) - optional GRPO training orchestration.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Optional: GPU stack and training dependencies for GRPO fine-tuning

### Installation

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn triagerl.api.server:app --host 0.0.0.0 --port 8000
```

You can also run the module directly:

```bash
python -m triagerl.api.server
```

### Minimal Environment Flow

Use the environment directly for local testing:

```python
from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.core.models import TriageAction

task_map, task_ids = load_all_tasks()
task = get_task(task_ids[0])
env = MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0)
obs = env.reset(seed=7)
action = TriageAction(
  action_type="classify",
  esi_level=int(task.esi_correct),
  reasoning="Clinical reasoning with enough detail to justify the triage decision.",
  recommended_actions=list(task.expected_actions or [])[:2],
  confidence=0.9,
)
next_obs, reward, done, info = env.step(action)
```

Minimal classify payload:

```json
{
  "action_type": "classify",
  "esi_level": 2,
  "reasoning": "Clinical reasoning with enough detail to justify the triage decision.",
  "recommended_actions": ["12-lead ECG", "IV access"],
  "confidence": 0.9
}
```

Minimal clarify action:

```json
{
  "action_type": "clarify",
  "clarifying_question": "Any chest pain, dyspnea, or syncope?",
  "reasoning": "I need one high-yield detail before classifying safely.",
  "recommended_actions": [],
  "confidence": 0.5
}
```

---

## Reward System

| Case | Expected Behavior |
|------|-------------------|
| **Good classification** | Highest reward |
| **Keyword stuffing** | Lower than good |
| **Clarify spam** | Slightly negative, not profitable |
| **Bad classification** | Strongly penalized |
| **Malformed output** | Explicit failure penalty |

The reward pipeline is intentionally conservative:
- `compute_final_score()` is the terminal grading step for valid classify actions.
- Clarify actions are not a profitable loop.
- The pipeline avoids hidden randomness during scoring.
- Invalid or malformed completions are rejected instead of being silently accepted.

---

## Validation and Testing

The repository is designed to be auditable with simple runtime checks.

### Suggested smoke checks

```bash
python -c "from triagerl.tasks.loader import load_all_tasks, get_task; from triagerl.env.triage_env import MedicalTriageEnv; from triagerl.core.models import TriageAction; task_map, task_ids = load_all_tasks(); task = get_task(task_ids[0]); env = MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0); obs = env.reset(seed=7); action = TriageAction(action_type='classify', esi_level=int(task.esi_correct), reasoning='Clinical reasoning with enough detail.', recommended_actions=list(task.expected_actions or [])[:2], confidence=0.9); print(env.step(action))"
```

### What to verify
- Reset works repeatedly without leakage.
- A valid classify step returns a finite reward.
- Clarify steps remain slightly negative.
- The same prompt/completion pair returns the same reward across repeated runs.

---

## Training Entry Point

`triagerl.training.train` provides the optional GRPO orchestration layer.

```bash
python -m triagerl.training.train
```

Notes:
- Training dependencies such as `trl` and `unsloth` may be required for full fine-tuning.
- The training loop is intended to refresh the dataset between epochs.
- The reward function is deterministic and reuses the shared grading pipeline.

---

## Roadmap

### Current State
- Deterministic prompt and reward pipeline.
- Episode replay for mid-episode prompts.
- Clarify exploit removed.
- On-policy dataset refresh support.

### Next Improvements
- Broader benchmark coverage across more triage scenarios.
- More formal regression tests for reward ordering.
- Extra runtime metrics around long-running training sessions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, issues, or hackathon submission support, use the repository issue tracker or the project maintainer’s contact details in your submission notes.
