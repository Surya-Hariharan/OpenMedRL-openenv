"""
Tests for triagerl.reward.grader

Covers: weight normalization, scoring formula, reward bounds, exploit scenarios.
"""
import pytest

from triagerl.reward.grader import (
    W_ESI, W_TEMPORAL, W_REASONING, W_ACTIONS, W_PATH,
    compute_final_score,
)
from triagerl.core.models import TriageAction


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

class TestWeights:
    def test_primary_weights_sum_to_one(self):
        total = W_ESI + W_TEMPORAL + W_REASONING + W_ACTIONS
        assert abs(total - 1.0) < 1e-9, (
            f"Primary weights must sum to exactly 1.0, got {total}. "
            f"W_ESI={W_ESI}, W_TEMPORAL={W_TEMPORAL}, "
            f"W_REASONING={W_REASONING}, W_ACTIONS={W_ACTIONS}"
        )

    def test_path_bonus_is_small(self):
        # W_PATH is outside the 1.0 budget — should be a small bonus only
        assert W_PATH <= 0.10, f"W_PATH={W_PATH} is too large; it should be a small bonus (≤0.10)"

    def test_esi_dominates(self):
        # ESI must be the largest weight for medical safety reasons
        assert W_ESI > W_TEMPORAL
        assert W_ESI > W_REASONING
        assert W_ESI > W_ACTIONS
        assert W_ESI > W_PATH


# ---------------------------------------------------------------------------
# compute_final_score bounds
# ---------------------------------------------------------------------------

_STEMI_TASK = {
    "id": "test-stemi",
    "esi_correct": 1,
    "expected_clarify_steps": 1,
    "key_reasoning_keywords": ["STEMI", "ECG", "troponin", "cath lab", "aspirin"],
    "expected_actions": ["12-lead ECG", "IV access", "activate cath lab", "aspirin", "oxygen"],
    "key_clarify_actions": ["check_vitals"],
    "hidden_info": [{"trigger": "check_vitals", "data": {}, "reveal_label": "repeat vitals", "clinical_priority": 1}],
    "max_steps": 3,
    "undertriage_consequence": "Death",
    "clinical_pearl": "Time is myocardium.",
    "vital_drift": {"per_step": {}, "noise_sigma": {}, "starts_at_step": 1},
    "initial_vitals": {"heart_rate": 110},
    "patient_info": {
        "patient_id": "test-stemi-001",
        "age": 65,
        "sex": "male",
        "symptoms": [],
        "medical_history": [],
        "medications": [],
        "allergies": [],
        "time_of_onset": "35 minutes ago, sudden onset",
    },
    "initial_labs": [],
    "confounders": [],
    "expected_severity": "critical",
    "difficulty": "easy",
    "category": "cardiovascular",
    "chief_complaint": "chest pain",
    "scenario": "Classic STEMI",
    "why_difficulty": "Textbook STEMI presentation requiring immediate cath lab activation.",
}


def _make_action(esi_level=1, reasoning="", recommended_actions=None, confidence=0.9):
    return TriageAction(
        action_type="classify",
        esi_level=esi_level,
        reasoning=reasoning or "Placeholder reasoning for testing purposes only.",
        recommended_actions=recommended_actions or [],
        confidence=confidence,
    )


class TestComputeFinalScore:
    def test_reward_always_in_bounds(self):
        action = _make_action(esi_level=1)
        score, _, _ = compute_final_score(action, _STEMI_TASK)
        assert -1.0 <= score <= 1.0

    def test_perfect_action_scores_high(self):
        reasoning = (
            "This patient has a classic STEMI presentation with acute chest pain and "
            "ST elevation on ECG. Troponin elevation confirms myocardial injury. "
            "Immediate cath lab activation with aspirin is indicated. The presentation "
            "is time-critical and requires ESI 1 classification. Oxygen, IV access, "
            "and continuous cardiac monitoring are all required urgently."
        )
        action = _make_action(
            esi_level=1,
            reasoning=reasoning,
            recommended_actions=["12-lead ECG", "IV access", "activate cath lab", "aspirin", "oxygen"],
            confidence=0.95,
        )
        score, components, _ = compute_final_score(action, _STEMI_TASK, steps_taken=2)
        assert score >= 0.60, f"Perfect action scored too low: {score}\nComponents: {components}"
        assert components["esi_score"] == 1.0

    def test_undertriage_critical_patient_penalised(self):
        action = _make_action(esi_level=3)  # ESI 1 patient classified as ESI 3
        score, components, _ = compute_final_score(action, _STEMI_TASK)
        assert score < 0, f"Undertriage of critical patient should score negative, got {score}"
        assert components["safety_modifier"] < 1.0

    def test_keyword_stuffing_scores_lower_than_honest(self):
        keywords = _STEMI_TASK["key_reasoning_keywords"]
        stuffed_reasoning = " | ".join(keywords * 8)
        stuffed_action = _make_action(esi_level=1, reasoning=stuffed_reasoning)

        honest_reasoning = (
            "The patient presents with classic STEMI. The 12-lead ECG confirms ST elevation. "
            "Troponin is elevated indicating myocardial injury. Cath lab should be activated now. "
            "Aspirin should be administered immediately. This is a time-critical ESI 1 emergency."
        )
        honest_action = _make_action(esi_level=1, reasoning=honest_reasoning)

        stuffed_score, _, _ = compute_final_score(stuffed_action, _STEMI_TASK, steps_taken=1)
        honest_score, _, _ = compute_final_score(honest_action, _STEMI_TASK, steps_taken=1)

        assert stuffed_score <= honest_score, (
            f"Keyword stuffing ({stuffed_score:.3f}) should not outscore "
            f"honest reasoning ({honest_score:.3f})"
        )

    def test_wrong_esi_on_non_critical_off_by_one(self):
        noncritical_task = {**_STEMI_TASK, "esi_correct": 4}
        action = _make_action(esi_level=3)  # off by one, not undertriage
        score, components, _ = compute_final_score(action, noncritical_task)
        # Safety modifier should not apply
        assert components["safety_modifier"] == 1.0
        # ESI score should be off-by-one (positive)
        assert components["esi_score"] > 0
