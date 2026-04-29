from __future__ import annotations

import json
import os
import re
from typing import List

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from triagerl.core.models import TriageAction
from triagerl.tasks.loader import get_task_list

load_dotenv()

# ── env vars with safe defaults ────────────────────────────────────────────────
BASE_URL     = os.getenv("BASE_URL",     "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "placeholder")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "triagerl"
MAX_STEPS    = 4

SYSTEM_PROMPT = """You are an experienced emergency department triage nurse with 15 years
of experience. You assess patients using the Emergency Severity Index:
ESI 1 = Immediate life-saving intervention or immediate life threat
ESI 2 = High risk situation, confused/lethargic/disoriented, or severe distress
ESI 3 = Urgent but stable, needs multiple resources
ESI 4 = Less urgent, needs one resource
ESI 5 = Non-urgent, no resources needed

Clinical safety policy:
- When in doubt, prioritize patient safety and choose higher acuity.
- Never undertriage high-risk presentations.
- Suspected sepsis with high-risk features must not be triaged below ESI 2.
- Hypotension, severe MI pattern, altered mental status with infection signs,
  or life-threatening presentation must be triaged as ESI 1.
- Non-blanching petechial rash with fever and neck stiffness = ESI 1 immediately.
  This is meningococcal septicemia until proven otherwise.
- Bacterial meningitis suspected = ESI 1. IV antibiotics within 1 hour is life-saving.
- Chest pain with hypotension AND diaphoresis AND arm radiation = ESI 1 STEMI pattern.
- Elderly patient with acute confusion, multiple comorbidities, abnormal medications:
  use clarify first to ask about urine appearance, recent temperature, and fluid output
  before classifying — these patients mask sepsis with normal-looking vitals.

Mandatory action rules:
- Always include minimum 4 recommended_actions for ESI 1 or ESI 2 cases.
- Chest pain ESI 1 must include: 12-lead ECG, aspirin 300mg, troponin, IV access.
- Meningitis ESI 1 must include: blood cultures, IV ceftriaxone, CT head, lumbar puncture.
- Sepsis ESI 2 must include: blood cultures, serum lactate, IV fluids, urine culture.
- Warfarin patient: always add check INR to recommended actions.
- Metformin patient with sepsis or AKI: always add hold metformin to recommended actions.

For each patient you must respond with valid JSON only, no markdown:
{
  "action_type": "classify" or "clarify",
  "esi_level": integer 1-5 (include if action_type is classify),
  "clarifying_question": "string" (include if action_type is clarify),
  "reasoning": "detailed clinical reasoning, minimum 50 words",
  "recommended_actions": ["action1", "action2", "action3", "action4"],
  "confidence": float 0.0 to 1.0
}

If you need more information before classifying, use clarify once.
For obvious emergencies, classify immediately."""


# ── helpers ────────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_action(text: str) -> TriageAction:
    cleaned = strip_code_fences(text)
    payload = json.loads(cleaned)
    return TriageAction.model_validate(payload)


def fallback_action() -> TriageAction:
    return TriageAction(
        action_type="classify",
        esi_level=3,
        reasoning="Fallback classification due to LLM parse error.",
        recommended_actions=[
            "immediate reassessment",
            "repeat full vital signs",
            "continuous monitoring",
            "urgent clinician review",
        ],
        confidence=0.0,
    )


def observation_to_prompt(observation: dict) -> str:
    return (
        "Patient observation:\n"
        f"{json.dumps(observation, indent=2)}\n\n"
        "Assess this patient carefully. Return JSON only — no markdown, "
        "no explanation outside the JSON object."
    )


def call_llm(client: OpenAI, observation: dict) -> TriageAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": observation_to_prompt(observation)},
        ],
        temperature=0,
        max_tokens=600,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty LLM response")
    return parse_action(content)


# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    task_id: str,
) -> tuple[bool, int, List[float]]:

    rewards: List[float] = []
    steps   = 0
    success = False

    try:
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as http:

            reset_resp = http.post("/reset", json={"task_id": task_id})
            reset_resp.raise_for_status()
            data        = reset_resp.json()
            observation = data["observation"]
            session_id  = data["info"]["session_id"]
            max_steps_for_episode = int(observation.get("max_steps", MAX_STEPS))

            while True:
                error_str = "null"
                action    = None

                try:
                    action = call_llm(client, observation)
                except Exception as exc:
                    error_str = str(exc).replace("\n", " ")[:200]
                    action    = fallback_action()

                action_payload = action.model_dump(exclude_none=True)
                action_str     = json.dumps(
                    action_payload, separators=(",", ":")
                )

                try:
                    step_resp = http.post(
                        "/step",
                        json={
                            "session_id": session_id,
                            "action": action_payload,
                        },
                    )
                    step_resp.raise_for_status()
                    result = step_resp.json()
                except Exception as exc:
                    error_str = str(exc).replace("\n", " ")[:200]
                    result    = {
                        "reward": 0.0,
                        "done":   True,
                        "observation": observation,
                    }

                reward = round(float(result.get("reward", 0.0)), 2)
                done   = bool(result.get("done", False))

                rewards.append(reward)
                steps += 1

                print(
                    f"[STEP] step={steps} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} "
                    f"error={error_str}",
                    flush=True,
                )

                observation = result.get("observation", observation)
                max_steps_for_episode = int(observation.get("max_steps", max_steps_for_episode))

                if done or steps >= max_steps_for_episode:
                    success = done and reward > 0.0
                    break

    except Exception as exc:
        error_str = str(exc).replace("\n", " ")[:200]
        if steps == 0:
            print(
                f"[STEP] step=1 action=null reward=0.00 "
                f"done=false error={error_str}",
                flush=True,
            )
            steps   = 1
            rewards = [0.0]

    return success, steps, rewards


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
    except Exception as exc:
        print(f"[WARN] LLM ping failed: {exc}", flush=True)

    task_ids = list(get_task_list())

    for task_id in task_ids:

        print(
            f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}",
            flush=True,
        )

        success, steps, rewards = run_episode(client, task_id)

        score       = rewards[-1] if rewards else 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()