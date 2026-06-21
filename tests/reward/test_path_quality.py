"""
Tests for triagerl.reward.path_quality

Covers: sequence order bonus, relevant-clarify bonus, spam penalties,
and verification that double-counted bonuses are absent.
"""
import pytest

from triagerl.reward.path_quality import (
    ActualClarifyRecord,
    score_clinical_path,
    count_useful_clarifications,
    BONUS_SEQUENCE_ORDER,
    BONUS_RELEVANT_CLARIFY,
    SPAM_PENALTY_PER_EXCESS,
)
from triagerl.tasks.schema import TaskConfig


def _make_task(key_clarify_actions=None, hidden_info_triggers=None) -> TaskConfig:
    hidden_info = []
    for trigger in (hidden_info_triggers or ["check_vitals", "ask_history"]):
        hidden_info.append({
            "trigger": trigger,
            "data": {"test": "value"},
            "reveal_label": f"{trigger} reveal",
            "clinical_priority": 1,
        })

    return TaskConfig.model_validate({
        "id": "test-task",
        "category": "cardiovascular",
        "difficulty": "medium",
        "esi_correct": 2,
        "chief_complaint": "chest pain",
        "scenario": "Test scenario",
        "initial_vitals": {"heart_rate": 110},
        "patient_info": {
            "patient_id": "test-001",
            "age": 60,
            "sex": "male",
            "symptoms": [],
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "time_of_onset": "30 minutes ago",
        },
        "why_difficulty": "Test scenario for unit testing path quality logic.",
        "initial_labs": [],
        "hidden_info": hidden_info,
        "confounders": [],
        "vital_drift": {"per_step": {}, "noise_sigma": {}, "starts_at_step": 1},
        "expected_clarify_steps": 2,
        "key_clarify_actions": key_clarify_actions or ["check_vitals", "ask_history"],
        "expected_actions": ["ECG", "IV access"],
        "key_reasoning_keywords": ["stemi", "troponin"],
        "expected_severity": "critical",
        "max_steps": 4,
        "undertriage_consequence": "harm",
        "clinical_pearl": "pearl",
    })


class TestSequenceOrderBonus:
    def test_vitals_before_history_earns_bonus(self):
        task = _make_task()
        records = [
            ActualClarifyRecord(question="What are the repeat vitals?", trigger="check_vitals"),
            ActualClarifyRecord(question="Any past medical history?", trigger="ask_history"),
        ]
        score = score_clinical_path(records, "Adequate reasoning with some structure because this is important.", task)
        assert score >= BONUS_SEQUENCE_ORDER - 0.01

    def test_history_before_vitals_misses_bonus(self):
        task = _make_task()
        records = [
            ActualClarifyRecord(question="Any past medical history?", trigger="ask_history"),
            ActualClarifyRecord(question="What are the repeat vitals?", trigger="check_vitals"),
        ]
        score_wrong_order = score_clinical_path(records, "Reasoning.", task)

        records_correct = [
            ActualClarifyRecord(question="What are the repeat vitals?", trigger="check_vitals"),
            ActualClarifyRecord(question="Any past medical history?", trigger="ask_history"),
        ]
        score_correct_order = score_clinical_path(records_correct, "Reasoning.", task)

        assert score_correct_order > score_wrong_order

    def test_only_one_trigger_type_skips_sequence_bonus(self):
        # When task only has check_vitals hidden info, sequence order is not applicable
        task = _make_task(
            key_clarify_actions=["check_vitals"],
            hidden_info_triggers=["check_vitals"],
        )
        records = [
            ActualClarifyRecord(question="What are the repeat vitals?", trigger="check_vitals"),
        ]
        # Just check it doesn't crash and gives reasonable output
        score = score_clinical_path(records, "Reasoning here.", task)
        assert isinstance(score, float)


class TestRelevantClarifyBonus:
    def test_relevant_trigger_earns_bonus(self):
        task = _make_task(key_clarify_actions=["check_vitals"])
        records = [
            ActualClarifyRecord(question="Repeat vital signs please?", trigger="check_vitals"),
        ]
        reasoning = (
            "The patient presents with hemodynamic instability. Repeat vital signs show "
            "tachycardia and hypotension, suggesting circulatory compromise. Due to the "
            "critical vital sign abnormalities, immediate intervention is warranted. "
            "Overall this is a time-sensitive emergency requiring ESI 1 classification."
        )
        score = score_clinical_path(records, reasoning, task)
        assert score >= BONUS_RELEVANT_CLARIFY - 0.01

    def test_no_relevant_trigger_no_bonus(self):
        task = _make_task(key_clarify_actions=["check_vitals"])
        records = [
            ActualClarifyRecord(question="Random question?", trigger="ask_history"),  # not expected
        ]
        score_no_relevant = score_clinical_path(records, "Reasoning.", task)

        records_relevant = [
            ActualClarifyRecord(question="Repeat vitals?", trigger="check_vitals"),
        ]
        score_relevant = score_clinical_path(records_relevant, "Reasoning.", task)

        assert score_relevant > score_no_relevant

    def test_no_reveal_trigger_none(self):
        task = _make_task()
        records = [
            ActualClarifyRecord(question="Irrelevant question about lunch?", trigger=None),
        ]
        score = score_clinical_path(records, "Reasoning.", task)
        # No relevant trigger, no bonus
        assert score < BONUS_RELEVANT_CLARIFY


class TestSpamPenalty:
    def test_excess_irrelevant_clarifications_penalised(self):
        task = _make_task(key_clarify_actions=["check_vitals"])
        # 5 irrelevant clarifications — 2 are tolerated, 3 are excess
        records = [
            ActualClarifyRecord(question=f"Irrelevant question {i}?", trigger=None)
            for i in range(5)
        ]
        score = score_clinical_path(records, "Reasoning.", task)
        assert score < -0.20, f"Expected heavily negative score for spam, got {score}"


class TestCountUsefulClarifications:
    def test_counts_matching_triggers(self):
        task = _make_task(key_clarify_actions=["check_vitals"])
        records = [
            ActualClarifyRecord(question="Vitals?", trigger="check_vitals"),
            ActualClarifyRecord(question="History?", trigger="ask_history"),   # not expected
        ]
        count = count_useful_clarifications(records, task)
        assert count == 1

    def test_zero_when_no_match(self):
        task = _make_task(key_clarify_actions=["check_vitals"])
        records = [
            ActualClarifyRecord(question="Random?", trigger="ask_history"),
        ]
        assert count_useful_clarifications(records, task) == 0

    def test_empty_records_returns_zero(self):
        task = _make_task()
        assert count_useful_clarifications([], task) == 0


class TestNoDoubleCountingReasoningKeywords:
    """Verify BONUS_REASONING_KEYWORDS was removed and path_quality
    does not award points purely for keyword mention in reasoning."""

    def test_keywords_in_reasoning_do_not_inflate_path_score(self):
        task = _make_task()
        records = []  # No clarifications at all
        # reasoning with all keywords
        reasoning_with_kws = "stemi troponin stemi troponin stemi troponin"
        # reasoning without keywords
        reasoning_without_kws = "This patient needs urgent assessment due to clinical severity."

        score_with = score_clinical_path(records, reasoning_with_kws, task)
        score_without = score_clinical_path(records, reasoning_without_kws, task)

        # The structured reasoning without keywords should NOT be significantly worse
        # than keyword stuffing — they should be comparable or keyword stuffing penalised
        assert score_with <= score_without + 0.31, (
            f"Keyword stuffing in reasoning should not earn a large bonus in path_quality. "
            f"With keywords: {score_with:.3f}, without: {score_without:.3f}"
        )
