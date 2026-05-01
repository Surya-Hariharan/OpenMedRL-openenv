from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path when running in containers
# (Hugging Face Spaces runs from /app and may not automatically make
# the repo discoverable). This mirrors adding the repo to PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Callable, List, Tuple

import gradio as gr

from triagerl.core.models import TriageAction
from triagerl.env.triage_env import MedicalTriageEnv
from triagerl.tasks import load_all_tasks


GLOBAL_SEED = 7
MAX_DEMO_STEPS = 5


TASK_MAP, TASK_IDS = load_all_tasks()
if not TASK_IDS:
    raise RuntimeError("No triage tasks were loaded.")

DEFAULT_TASK_ID = TASK_IDS[0]


def _create_env() -> MedicalTriageEnv:
    task = TASK_MAP[DEFAULT_TASK_ID]
    return MedicalTriageEnv(
        task_config=task,
        seed=GLOBAL_SEED,
        clarify_penalty_dampening=1.0,
    )


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def _format_vitals(obs) -> str:
    vitals = obs.patient.vitals
    parts = [
        f"HR {_format_number(vitals.heart_rate)}",
        f"BP {_format_number(vitals.blood_pressure_systolic)}/{_format_number(vitals.blood_pressure_diastolic)}",
        f"RR {_format_number(vitals.respiratory_rate)}",
        f"SpO2 {_format_number(vitals.oxygen_saturation)}",
        f"Temp {_format_number(vitals.temperature)}",
        f"GCS {_format_number(vitals.gcs)}",
    ]
    return ", ".join(parts)


def _summarize_observation(obs) -> str:
    symptoms = ", ".join(obs.patient.symptoms[:4]) if obs.patient.symptoms else "none listed"
    history = ", ".join(obs.patient.medical_history[:3]) if obs.patient.medical_history else "none listed"
    meds = ", ".join(obs.patient.current_medications[:3]) if obs.patient.current_medications else "none listed"
    allergies = ", ".join(obs.patient.allergies[:3]) if obs.patient.allergies else "none listed"
    if obs.revealed_info:
        revealed_items = ", ".join(f"{key}: {value}" for key, value in list(obs.revealed_info.items())[:3])
        additional = revealed_items
    else:
        additional = "revealed" if obs.additional_info_revealed else "none"
    return (
        f"Chief complaint: {obs.patient.chief_complaint}\n"
        f"Symptoms: {symptoms}\n"
        f"Vitals: {_format_vitals(obs)}\n"
        f"History: {history}\n"
        f"Meds: {meds}\n"
        f"Allergies: {allergies}\n"
        f"Additional info: {additional}"
    )


def _targeted_question(task) -> str:
    complaint = f"{task.chief_complaint} {task.scenario}".lower()
    if any(term in complaint for term in ("chest", "cardiac", "heart")):
        return "Can you describe the chest pain severity, radiation, and any associated shortness of breath or sweating?"
    if any(term in complaint for term in ("abdominal", "belly", "stomach")):
        return "Can you clarify the pain location, migration, vomiting, fever, or guarding?"
    if any(term in complaint for term in ("breath", "respir", "asthma", "cough")):
        return "How severe is the breathing problem, and is there wheezing, cough, fever, or chest tightness?"
    if any(term in complaint for term in ("head", "neuro", "seiz", "stroke", "weakness")):
        return "Any new weakness, speech change, confusion, seizure activity, or sudden onset?"
    return "What important symptoms, triggers, and recent changes should I know to triage this case safely?"


def _build_classify_action(task, *, esi_level: int, reasoning: str, recommended_actions: List[str], confidence: float) -> TriageAction:
    return TriageAction(
        action_type="classify",
        esi_level=esi_level,
        reasoning=reasoning,
        recommended_actions=recommended_actions,
        confidence=confidence,
    )


def _build_clarify_action(question: str, reasoning: str, confidence: float) -> TriageAction:
    return TriageAction(
        action_type="clarify",
        clarifying_question=question,
        reasoning=reasoning,
        confidence=confidence,
    )


def _run_episode(policy_name: str, action_plan: Callable[[object, object], List[TriageAction]]) -> str:
    env = _create_env()
    obs = env.reset(seed=GLOBAL_SEED)
    task = env.task_config

    actions = action_plan(task, obs)
    lines = [f"=== Policy: {policy_name} ===", f"Task: {task.id}", "Observation Summary:", _summarize_observation(obs), "Actions:"]

    final_reward = 0.0
    final_done = False

    for index, action in enumerate(actions[:MAX_DEMO_STEPS], start=1):
        obs, reward, done, info = env.step(action)
        final_reward = reward
        final_done = done
        action_desc = action.model_dump(exclude_none=True)
        lines.append(f"{index}. {action_desc}")
        lines.append(f"   Step reward: {reward:.2f} | done: {done} | phase: {info.get('workflow_phase', 'unknown')}")
        if done:
            break

    if not final_done and env.current_step >= MAX_DEMO_STEPS:
        lines.append(f"Stopped after {MAX_DEMO_STEPS} steps to keep the demo bounded.")

    lines.append(f"Final reward: {final_reward:.2f}")
    lines.append("")
    lines.append(f"Final observation summary:\n{_summarize_observation(obs)}")
    return "\n".join(lines)


def run_good_policy() -> str:
    def plan(task, obs) -> List[TriageAction]:
        actions: List[TriageAction] = []
        clarify_steps = 1 if task.expected_clarify_steps > 0 else 0
        clarify_steps = min(clarify_steps, MAX_DEMO_STEPS - 1)

        if clarify_steps:
            actions.append(
                _build_clarify_action(
                    _targeted_question(task),
                    "Gather one high-yield detail before disposition.",
                    0.85,
                )
            )

        actions.append(
            _build_classify_action(
                task,
                esi_level=int(task.esi_correct),
                reasoning="Focused triage reasoning that matches the presentation, urgency, and expected disposition.",
                recommended_actions=list(task.expected_actions[:3]),
                confidence=0.92,
            )
        )
        return actions

    return _run_episode("Good Policy", plan)


def run_bad_policy() -> str:
    def plan(task, obs) -> List[TriageAction]:
        wrong_esi = 1 if int(task.esi_correct) != 1 else 5
        return [
            _build_classify_action(
                task,
                esi_level=wrong_esi,
                reasoning="Not enough clinical detail, but classifying immediately with minimal justification.",
                recommended_actions=[],
                confidence=0.20,
            )
        ]

    return _run_episode("Bad Policy", plan)


def run_keyword_policy() -> str:
    def plan(task, obs) -> List[TriageAction]:
        keywords = list(task.key_reasoning_keywords[:8])
        filler = ["triage", "risk", "safety", "assessment", "urgency", "disposition"]
        dense_reasoning = " | ".join(keywords + filler + keywords + filler)
        return [
            _build_clarify_action(
                "Can you repeat the same details in a more elaborate way?",
                "Collecting extra text before disposition.",
                0.40,
            ),
            _build_classify_action(
                task,
                esi_level=int(task.esi_correct),
                reasoning=dense_reasoning,
                recommended_actions=["monitor", "escalate", "reassess"],
                confidence=0.55,
            ),
        ]

    return _run_episode("Keyword Stuffing", plan)


def run_simulation(policy_name: str) -> str:
    if policy_name == "Good Policy":
        return run_good_policy()
    if policy_name == "Bad Policy":
        return run_bad_policy()
    if policy_name == "Keyword Stuffing":
        return run_keyword_policy()
    return "Unknown policy selected."


with gr.Blocks(title="TriageRL Demo", analytics_enabled=False) as demo:
    gr.Markdown("# TriageRL Demo\nA small Gradio wrapper around the deterministic triage environment.")
    policy = gr.Dropdown(
        choices=["Good Policy", "Bad Policy", "Keyword Stuffing"],
        value="Good Policy",
        label="Policy",
    )
    run_button = gr.Button("Run Simulation", variant="primary")
    output = gr.Textbox(label="Simulation Output", lines=24, max_lines=40)

    run_button.click(fn=run_simulation, inputs=policy, outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)