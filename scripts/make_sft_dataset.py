from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from medical_triage_env.tasks import load_all_tasks


def paraphrase(text: str, rng: random.Random) -> str:
    """Cheap, deterministic-ish paraphraser (no external LLM)."""
    t = (text or "").strip()
    if not t:
        return t
    swaps: List[Tuple[str, List[str]]] = [
        ("sudden", ["abrupt", "acute", "sudden"]),
        ("severe", ["severe", "intense", "marked"]),
        ("short of breath", ["short of breath", "breathless", "dyspnoeic"]),
        ("worse", ["worse", "worsening", "progressively worse"]),
        ("pain", ["pain", "discomfort"]),
        ("fever", ["fever", "high temperature"]),
        ("confused", ["confused", "disoriented", "not herself"]),
        ("vomiting", ["vomiting", "retching"]),
    ]
    out = t
    for needle, opts in swaps:
        if needle in out.lower():
            choice = rng.choice(opts)
            # naive replace preserving original casing mostly
            out = out.replace(needle, choice).replace(needle.title(), choice.title())
    return out


def build_prompt(task: Any, variant: int, rng: random.Random) -> str:
    cfg = task
    chief = paraphrase(cfg.chief_complaint, rng)
    scenario = paraphrase(cfg.scenario, rng)
    symptoms = ", ".join(cfg.patient_info.symptoms[:8])
    hx = ", ".join(cfg.patient_info.medical_history[:6])
    meds = ", ".join(cfg.patient_info.current_medications[:6])
    vitals = cfg.initial_vitals

    # Simple "triage note" prompt format that works well for demos.
    return (
        "You are an ED triage model. Decide whether to ask ONE clarifying question "
        "or assign an ESI (1-5) with brief reasoning.\n\n"
        f"Case (variant {variant}): {scenario}\n"
        f"Chief complaint: {chief}\n"
        f"Age/Sex: {cfg.patient_info.age} / {cfg.patient_info.sex}\n"
        f"Symptoms: {symptoms}\n"
        f"History: {hx}\n"
        f"Medications: {meds}\n"
        f"Initial vitals: {json.dumps(vitals, ensure_ascii=True)}\n"
    )


def expert_trajectory(task: Any) -> List[Dict[str, Any]]:
    """
    Produce a tiny supervised trajectory:
    - 0..expected_clarify_steps clarifies (based on key_clarify_actions)
    - final classify with correct ESI and templated reasoning
    """
    steps: List[Dict[str, Any]] = []

    keys = list(task.key_clarify_actions or [])
    # Ensure at least one good clarify if expected_clarify_steps > 0.
    desired = int(task.expected_clarify_steps or 0)

    clarify_templates = {
        "ask_history": "Any key history details, symptom progression, and medication adherence (incl. anticoagulants)?",
        "check_vitals": "Please provide repeat vitals (BP both arms if relevant), oxygen saturation trend, and GCS.",
        "examine_patient": "What are the focused physical exam findings (resp/cardiac/abd/neuro) and any red flags?",
        "clarify": "What additional high-yield information would change triage urgency right now?",
    }

    for i in range(desired):
        key = keys[i] if i < len(keys) else "ask_history"
        q = clarify_templates.get(key, clarify_templates["clarify"])
        steps.append(
            {
                "action_type": "clarify",
                "clarifying_question": q,
                "reasoning": "Gather high-yield missing information to reduce unsafe triage error.",
                "recommended_actions": [],
                "confidence": 0.45,
            }
        )

    steps.append(
        {
            "action_type": "classify",
            "esi_level": int(task.esi_correct),
            "reasoning": (
                f"Assign ESI {int(task.esi_correct)} based on acuity, vitals, red flags, and time-critical risk. "
                f"Key considerations: {', '.join((task.key_reasoning_keywords or [])[:6])}."
            ),
            "recommended_actions": list((task.expected_actions or [])[:8]),
            "confidence": 0.65,
        }
    )

    return steps


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create small SFT dataset from task corpus")
    p.add_argument("--out", type=str, default="data/sft.jsonl")
    p.add_argument("--variants-per-task", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tasks, task_ids = load_all_tasks()
    rows: List[Dict[str, Any]] = []

    for tid in task_ids:
        t = tasks[tid]
        traj = expert_trajectory(t)
        for v in range(args.variants_per_task):
            prompt = build_prompt(t, variant=v, rng=rng)
            # Emit step-wise SFT examples (prompt + action json).
            for step_idx, action in enumerate(traj):
                rows.append(
                    {
                        "task_id": tid,
                        "variant": v,
                        "step_idx": step_idx,
                        "prompt": prompt,
                        "response_json": action,
                    }
                )

    write_jsonl(Path(args.out), rows)
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()

