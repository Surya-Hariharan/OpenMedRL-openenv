# Medical Triage Environment

An OpenEnv-style emergency department triage benchmark built around the Emergency Severity Index (ESI 1-5). The agent receives a structured patient presentation and must either classify urgency or ask a focused clarifying question when more information is needed.

This repo includes 20 clinical scenarios across cardiovascular, neurological, infectious, respiratory, and abdominal/metabolic cases. The environment adds partial observability, stochastic vital drift, deterministic scoring, and safety-focused undertriage penalties.

## What’s Included

- FastAPI environment with `/reset`, `/step`, `/state`, `/tasks`, and `/health` endpoints.
- Structured observations and actions for triage-style LLM agents.
- Episode grading with reward shaping, safety modifiers, and telemetry.
- A deterministic baseline evaluator for train/test splits.
- A small JSONL dataset generator for SFT-style experiments.
- A smoke-test script for publishing and API redaction checks.

## Quick Start

### Local install

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn triagerl.api.server:app --host 0.0.0.0 --port 8000
```

You can also use the project script defined in `pyproject.toml`:

```bash
python -m server.app
```

### Run with Docker

```bash
docker build -t medical-triage-env .
docker run -p 8000:8000 medical-triage-env
```

## API Workflow

1. `POST /reset` to start a new episode.
2. `POST /step` with a `session_id` and a `TriageAction` payload.
3. `GET /state?session_id=...` to inspect the current episode state.
4. `GET /tasks` to list tasks only when `ENV=development`; production mode redacts the list.

Minimal action schema:

```json
{
	"action_type": "classify",
	"esi_level": 2,
	"reasoning": "Clinical reasoning text",
	"recommended_actions": ["12-lead ECG", "IV access"],
	"confidence": 0.8
}
```

For a clarification step, set `action_type` to `clarify` and provide `clarifying_question` instead of `esi_level`.

## Example Episode

```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id":"classic-stemi"}'
```

The response includes an opaque `session_id`. Use that value in subsequent `/step` requests:

```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{
	"session_id": "<session_id>",
	"action": {
		"action_type": "clarify",
		"clarifying_question": "Any relevant history, medications, or repeat vitals?",
		"reasoning": "I need high-yield missing information before classifying.",
		"recommended_actions": [],
		"confidence": 0.45
	}
}'
```

## Session Storage

Sessions are kept in memory by default. If `REDIS_URL` is set, the environment switches to Redis-backed session storage so episodes can survive multi-worker deployments.

## Project Layout

- `triagerl/api/server.py` - FastAPI app entrypoint.
- `triagerl/env/triage_env.py` - episode orchestration logic.
- `triagerl/tasks/` - task corpus and validation models.
- `triagerl/reward/` - reward and grading logic.
- `triagerl/env/revealer.py` - partial observability and vital drift.
- `triagerl/api/session/` - session store implementation.
- The submission bundle intentionally excludes local training, evaluation, and debug scripts.

## Notes

- `openenv.yaml` defines the OpenEnv metadata for the benchmark.
- Set `ENV=development` for local debugging if you want task identifiers exposed in development-only paths.
- Keep secrets out of version control; use `.env` locally when needed.
