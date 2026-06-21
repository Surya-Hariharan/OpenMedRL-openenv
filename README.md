# TriageRL

**A deterministic reinforcement learning environment for emergency department triage.**

TriageRL is an OpenEnv-compatible benchmark where LLM agents must classify patient urgency using the Emergency Severity Index (ESI 1–5), gather clinical information under partial observability, and reason safely under time pressure. Every reward is deterministic: the same prompt and completion always produce the same score.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-black)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-0.2.0-green)

---

## Overview

The agent observes a partially observable patient presentation and must choose between:

- **Clarify** — ask a focused clinical question to reveal hidden information (vitals, history, examination findings).
- **Classify** — assign an ESI level (1–5) and provide clinical reasoning and recommended interventions.

The environment rewards correct, efficient, and well-reasoned classification. It penalises undertriage of critical patients, keyword stuffing, clarification spam, and malformed outputs.

---

## Task Corpus

**35 hand-authored clinical scenarios** across **9 categories** and **3 difficulty tiers.**

| Category | Tasks | Difficulty range |
|---|---|---|
| Cardiovascular | 4 | Easy → Hard |
| Neurological | 4 | Medium → Hard |
| Infectious / Sepsis | 4 | Hard |
| Respiratory | 4 | Easy → Hard |
| Abdominal / Metabolic | 4 | Easy → Hard |
| Pediatric | 4 | Easy → Hard |
| Toxicology | 4 | Easy → Hard |
| Obstetric | 3 | Easy → Medium |
| Trauma | 4 | Medium → Hard |

Representative tasks:
- `classic-stemi` — STEMI with shock physiology (ESI 1, Easy)
- `aortic-dissection-mimic` — Type A dissection presenting as back pain (ESI 1, Hard)
- `acute-asthma-silent-chest` — Life-threatening asthma; silent chest is a late sign (ESI 1, Medium)
- `masked-urosepsis` — Atypical elderly sepsis masked by beta-blockade (ESI 2, Hard)
- `tricyclic-antidepressant-od` — TCA overdose; QRS widening is the danger sign (ESI 1, Medium)
- `stroke-lacunar-subtle` — Ischaemic stroke within thrombolysis window on anticoagulation (ESI 2, Medium)

---

## Reward System

All scoring is deterministic and pure: no I/O, no randomness, same inputs always produce the same output.

### Terminal reward (on CLASSIFY)

```
base = 0.68 × esi_accuracy        # [-0.15, 1.00]  — primary signal
     + 0.10 × temporal_efficiency  # [0.00,  1.00]  — urgency-aware speed
     + 0.12 × reasoning_quality    # [0.00,  1.00]  — keyword coverage + anti-gaming
     + 0.10 × action_coverage      # [0.00,  1.00]  — expected interventions covered
     + 0.05 × path_quality         # [0.00,  1.00]  — bonus for correct workflow sequence

if undertriage (ESI ≤2 patient classified higher):
    final = clamp(base × 0.25, −1, 1)   # sign-aware penalty amplification
else:
    final = clamp(base, −1, 1)
```

**ESI accuracy scoring:**

| Prediction vs ground truth | Score |
|---|---|
| Correct | +1.00 |
| Off by 1 (non-critical) | +0.30 |
| Off by 1 (critical patient, undertriage) | −0.75 |
| Off by 2 | −0.50 |
| Off by 3+ | −1.00 |

### Per-step shaping (on CLARIFY)

Clarify actions are never profitable in isolation. Rewards are slightly negative to prevent loops:

| Condition | Shaping reward |
|---|---|
| Relevant trigger revealed | −0.01 |
| Irrelevant or no reveal | −0.02 to −0.03 |
| Hard over-budget clarify (with dampening) | −0.05, episode may terminate |

### Reward ordering guarantee

```
correct ESI + targeted clarify > correct ESI direct > keyword-stuffed > wrong ESI > undertriage critical
```

---

## Environment Design

### Partial Observability

Initial observations expose the chief complaint, patient demographics, arrival mode, and a subset of vital signs. Hidden information is revealed only when the agent asks the right type of question:

| Trigger | Reveals | Example keywords |
|---|---|---|
| `ask_history` | Medical history, medications, allergies, adherence | "medications", "past medical history", "allergies" |
| `check_vitals` | Repeat vitals: BP, HR, SpO₂, GCS, ECG findings | "blood pressure", "oxygen saturation", "heart rate" |
| `examine_patient` | Physical examination: auscultation, inspection, red-flag signs | "examine", "auscultate", "palpate", "inspection" |

Trigger assignment uses keyword-length scoring: longer, more specific clinical terms score higher. Each trigger can be revealed at most once per episode.

### Vital Sign Drift

Patient vitals evolve stochastically between steps using a configured per-vital Gaussian drift. A `deterioration_signal` (0.0–1.0) is computed from visible vitals and included in every observation as an urgency hint.

### Phase State Machine

```
ASSESSMENT  →(clarify, budget ok)→   ASSESSMENT
ASSESSMENT  →(clarify, exhausted)→   DISPOSITION
ASSESSMENT  →(classify)→             COMPLETED
DISPOSITION →(clarify)→              DISPOSITION   (penalised)
DISPOSITION →(classify)→             COMPLETED
COMPLETED   →(any)→                  InvalidTransitionError
```

### Curriculum

Training uses a three-phase curriculum across 20 epochs:

| Epochs | Difficulty weights | Clarify penalty dampening |
|---|---|---|
| 0–4 (Warm-up) | Easy 60%, Medium 30%, Hard 10% | 0.0 (free exploration) |
| 5–14 (Balanced) | Easy 30%, Medium 40%, Hard 30% | Linear ramp 0.0 → 1.0 |
| 15–19 (Advanced) | Easy 20%, Medium 40%, Hard 40% | 1.0 (full penalties) |

The `clarify_penalty_dampening` ramp prevents premature episode termination from killing exploration diversity during early GRPO training.

---

## Architecture

```
triagerl/
├── core/
│   ├── types.py          — PhaseState, ActionType, DifficultyTier, ClinicalCategory (enums)
│   ├── models.py         — VitalSigns, TriageAction, TriageObservation, EpisodeMetrics (Pydantic v2)
│   └── constants.py      — 350+ clinical keyword → trigger mappings
├── env/
│   ├── triage_env.py     — MedicalTriageEnv: episode orchestration, RewardCallback protocol
│   ├── episode.py        — EpisodeState: mutable episode state dataclass
│   ├── phase.py          — PhaseStateMachine: FSM with InvalidTransitionError
│   ├── drift.py          — VitalDriftEngine: Gaussian drift + deterioration signal
│   └── revealer.py       — InfoRevealer: keyword-scored progressive disclosure
├── reward/
│   ├── grader.py         — compute_final_score(): 5-component weighted scoring
│   ├── components.py     — score_esi, score_temporal, score_reasoning, score_actions
│   ├── path_quality.py   — score_clinical_path(): workflow sequence scoring
│   ├── safety.py         — apply_safety_modifier(): sign-aware undertriage penalty
│   ├── shaping.py        — compute_clarify_shaping(): per-step shaping rewards
│   └── llm_judge.py      — Async Anthropic API judge for offline evaluation
├── training/
│   ├── train.py          — GRPO orchestrator: 20 epochs, dampening ramp, on-policy refresh
│   ├── dataset.py        — TriageEpisodeDataset: curriculum sampling, mid-episode prompts
│   └── reward_fn.py      — triage_grpo_reward(): deterministic episode replay for GRPO
├── tasks/
│   ├── schema.py         — TaskConfig, HiddenInfoItem, VitalDrift, LabResult (Pydantic v2)
│   ├── loader.py         — Singleton task loader with curriculum batch sampling
│   └── corpus/tasks.yaml — 35 clinical scenarios (YAML)
├── api/
│   ├── server.py         — FastAPI: /reset /step /health /tasks
│   └── session/          — InMemorySessionStore + RedisSessionStore (TTL-eviction)
└── eval/
    └── client.py         — Evaluation client for batch episode runs
```

### Data flow

```
TaskConfig (YAML)
      │
      ▼
MedicalTriageEnv.reset(seed)
  ├── VitalDriftEngine.reset()
  ├── InfoRevealer.reset()
  ├── PhaseStateMachine.reset()
  └── EpisodeState.reset()
      │
      │  TriageObservation
      ▼
  LLM Agent
      │  TriageAction {clarify | classify}
      ▼
MedicalTriageEnv.step(action)
  ├── CLARIFY → InfoRevealer.process_clarify()  →  shaping reward
  │             VitalDriftEngine.apply()
  └── CLASSIFY → compute_final_score()           →  terminal reward
                 build_episode_metrics()
      │
      ▼
(obs, reward, done, info)
      │
      ▼
GRPO trainer  →  policy gradient update
```

---

## Quick Start

### Requirements

- Python 3.11+
- For GRPO training: `unsloth`, `trl` (GPU required)

### Install

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn triagerl.api.server:app --host 0.0.0.0 --port 8000
```

### Use the environment directly

```python
from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.core.models import TriageAction

task_map, task_ids = load_all_tasks()
task = get_task("classic-stemi")
env = MedicalTriageEnv(task_config=task, clarify_penalty_dampening=1.0)

obs = env.reset(seed=42)
print(f"Step {obs.step_number} | Phase: {obs.phase} | Deterioration: {obs.deterioration_signal:.2f}")

# Gather information first
clarify_action = TriageAction(
    action_type="clarify",
    clarifying_question="What are the patient's blood pressure and oxygen saturation?",
    reasoning="Need to assess haemodynamic stability before classifying.",
    confidence=0.6,
)
obs, reward, done, info = env.step(clarify_action)
print(f"Shaping reward: {reward:.3f} | Revealed: {obs.additional_info_revealed}")

# Then classify
classify_action = TriageAction(
    action_type="classify",
    esi_level=1,
    reasoning=(
        "Classic STEMI presentation: crushing chest pain radiating to left arm, "
        "diaphoresis, ST elevation on ECG. Haemodynamically unstable. "
        "Requires immediate cath lab activation, aspirin, IV access, and monitoring."
    ),
    recommended_actions=["12-lead ECG", "IV access", "aspirin 300mg", "activate cath lab", "oxygen"],
    confidence=0.95,
)
obs, reward, done, info = env.step(classify_action)
print(f"Terminal reward: {reward:.3f} | Done: {done}")
print(f"ESI correct: {info['episode_metrics'].esi_correct} | Predicted: {info['episode_metrics'].esi_predicted}")
```

### API usage

**Reset an episode:**
```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classic-stemi"}'
```

**Step with a classify action:**
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id_from_reset>",
    "action": {
      "action_type": "classify",
      "esi_level": 1,
      "reasoning": "STEMI with haemodynamic instability.",
      "recommended_actions": ["ECG", "cath lab"],
      "confidence": 0.9
    }
  }'
```

---

## Training

TriageRL v0.2 is configured to fine-tune Llama-3.1-8B via GRPO using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl).

### Configuration (v0.2.0)

| Parameter | Value |
|---|---|
| Base model | `unsloth/Llama-3.1-8B-Instruct-bnb-4bit` |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Samples per epoch | 512 |
| Epochs | 20 |
| Total steps | 2000 |
| GRPO generations | 8 |
| Effective batch size | 64 |
| Max sequence length | 3072 |
| Learning rate | 5e-6 |

### Run training

```bash
python -m triagerl.training.train
```

Key design decisions in the training pipeline:

- **Task ID embedded in prompt** — the reward function extracts the task ID from the prompt text, not from external state, guaranteeing that the same prompt always scores identically regardless of batch order.
- **On-policy dataset refresh** — `TriageEpisodeDataset.refresh(epoch)` regenerates all prompts at the start of each epoch using the current epoch's dampening level.
- **Mid-episode sampling** — 50% of training prompts represent mid-episode states where the agent has already asked one clarifying question. This forces the policy to learn from both first-step and second-step decisions.
- **Deterministic episode replay** — `_replay_to_step()` in `reward_fn.py` reconstructs any mid-episode state from the prompt context alone, with no shared mutable environment.

---

## Evaluation

The `EpisodeMetrics` object returned in `info["episode_metrics"]` at episode end contains:

```python
EpisodeMetrics(
    task_id="classic-stemi",
    difficulty="easy",
    category="cardiovascular",
    esi_correct=1,
    esi_predicted=1,
    steps_taken=2,
    max_steps=4,
    total_reward=0.847,
    undertriage=False,
    overtriage=False,
    clarification_count=1,
    useful_clarification_count=1,
    agent_confidence=0.95,
    reward_breakdown=RewardBreakdown(
        esi_accuracy=1.0,
        temporal_efficiency=0.88,
        reasoning_quality=0.91,
        action_coverage=1.0,
        path_quality=0.75,
        safety_modifier=1.0,
        final_reward=0.847,
    ),
)
```

For offline reasoning quality evaluation, `triagerl/reward/llm_judge.py` provides an async Anthropic API judge that scores reasoning across four dimensions (clinical accuracy, safety awareness, communication clarity, and decision justification).

---

## Exploit Resistance

The reward system includes explicit defences against common gaming strategies:

| Strategy | Defence |
|---|---|
| Keyword stuffing (dense clinical jargon) | Vocabulary diversity check: unique token ratio < 0.35 → 0.70× penalty |
| Extreme repetition | If ≥3 tokens appear ≥6 times → 0.80× penalty |
| Direct trigger injection ("check_vitals please") | −0.005 penalty per clarify step; detected by token presence |
| Clarify loop (asking indefinitely) | Hard budget: episode terminates if clarify_count > max(expected+2, 4) with dampening ≥ 0.3 |
| Confidence gaming (always 0.5) | Confidence is collected; calibration scoring is planned (see Roadmap) |

---

## Roadmap

The following gaps have been identified and blueprinted for implementation. See the [implementation blueprint](docs/blueprint.md) for exact class hierarchies, function signatures, and migration strategies.

### Gap 1 — Task Corpus Expansion (35 → 200+ tasks)

Expand to 200–500 validated clinical scenarios with stratified train/val/test/hidden splits, physician review metadata per task, and a `TaskGenerator` pipeline for LLM-assisted scenario drafting.

### Gap 2 — Disease-State Simulation Engine

Replace Gaussian vital drift with physiologically grounded disease-state machines (`SepsisModel`, `STEMIModel`, `StrokeModel`, `AsthmaModel`, `TraumaModel`, `PneumoniaModel`). Each model defines explicit state transitions (e.g., SIRS → sepsis → septic shock → MODS) with per-state vital trajectories and hard physiological clamps.

### Gap 3 — Environment Test Coverage

Add integration and unit tests for the environment layer (`triage_env.py`, `revealer.py`, `drift.py`, `phase.py`), which currently have no test coverage. Target 75% overall and 95% on the phase FSM.

### Gap 4 — Confidence Calibration Reward

Reward the `confidence` field that agents already produce. Add `score_uncertainty_calibration()` to the terminal reward pipeline: overconfident wrong predictions are penalised; appropriately uncertain requests for more information are rewarded. Track ECE and Brier Score in evaluation.

### Gap 5 — Expanded Action Space

Replace the coarse `CLARIFY` action with 13 typed request actions (`REQUEST_BP`, `REQUEST_HR`, `REQUEST_SPO2`, `REQUEST_LABS`, `REQUEST_IMAGING`, etc.) that each reveal only their specific information. Add `ESCALATE_SPECIALIST`, `MONITOR`, and `DISCHARGE` as explicit disposition actions with associated reward shaping.

---

## Project Structure

```
triagerl-openenv/
├── triagerl/                 — main package
│   ├── api/                  — FastAPI server and session management
│   ├── core/                 — Pydantic models, enums, clinical constants
│   ├── env/                  — episode simulation and environment orchestration
│   ├── eval/                 — evaluation client
│   ├── logs/                 — structured logging utilities
│   ├── reward/               — deterministic multi-component reward pipeline
│   ├── tasks/                — task schema, loader, and YAML corpus
│   └── training/             — GRPO dataset, reward function, and training loop
├── tests/                    — reward layer tests
├── app.py                    — Gradio demo (policy comparison)
├── openenv.yaml              — benchmark specification
├── pyproject.toml
└── requirements.txt
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Data validation | Pydantic v2 |
| API server | FastAPI + Uvicorn |
| Session storage | In-memory (default) or Redis |
| Training | TRL (GRPO) + Unsloth (LoRA) |
| Containerisation | Docker (Python 3.11, non-root user) |

---

## License

[MIT License](LICENSE)
