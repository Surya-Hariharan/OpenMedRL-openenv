---
<<<<<<< HEAD
title: medical-triage-env
sdk: docker
app_port: 8000
colorFrom: blue
colorTo: green
---

## Medical Triage Environment

An OpenEnv-style environment for emergency department triage using the Emergency Severity Index (ESI 1-5).

## Overview
The agent receives structured patient presentations and must either classify urgency or request a clarifying question when additional history is needed. The benchmark emphasizes triage prioritization, clinical reasoning, and safe escalation.

## What you can realistically do (hackathon-ready)
You likely *won’t* train a “frontier” model on ~20 cases. You *can* win a hackathon by showing:
- a working interactive API (`/reset`, `/step`)
- a repeatable eval split with safety metrics (undertriage rate)
- cheap text augmentation + an SFT-style dataset generator (no extra corpus required)

## Reward Summary
- ESI accuracy: 50%
- Reasoning quality: 30%
- Action appropriateness: 20%
- Undertriage penalty: applied for dangerous low-acuity assignments
- Urgency bonus: correct ESI on early steps
- Step penalty: small penalty per additional step

## Setup
```bash
pip install -r requirements.txt
docker build -t medical-triage-env .
docker run -p 8000:8000 medical-triage-env
openenv validate
python inference.py
```

## Hackathon scripts
Generate a small supervised dataset (JSONL) from the existing task corpus:

```bash
python scripts/make_sft_dataset.py --out data/sft.jsonl --variants-per-task 8
```

Run a deterministic baseline eval on a held-out split:

```bash
python scripts/eval_split.py --test-n 5 --episodes-per-task 3
```

Pre-publish smoke test (run with the server already running on `localhost:8000`):

```bash
python scripts/smoke_publish.py
```

## Environment Variables
- `API_BASE_URL`
- `API_KEY`
- `MODEL_NAME`
- `BASE_URL`

Use `.env` only for local development. Do not commit secrets.

## Baseline Scores
Run `python scripts/eval_split.py` to print baseline metrics for your current code + policy.
=======
title: Meta123
emoji: 🌍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> c453033df7ec6d4d68bf5ef21b9dac69152cb2b8
