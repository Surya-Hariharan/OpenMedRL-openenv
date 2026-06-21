"""
Tests for triagerl.reward.components

Covers: ESI scoring edge cases, temporal score, reasoning quality anti-gaming,
action coverage (majority-token matching), and keyword matching (all-token tier-2).
"""
import pytest

from triagerl.reward.components import (
    action_overlap,
    keyword_matches,
    score_actions,
    score_esi,
    score_reasoning,
    score_temporal,
    ESI_SCORE_PERFECT,
    ESI_SCORE_OFF_BY_ONE,
    ESI_SCORE_OFF_BY_TWO,
    ESI_SCORE_LARGE_MISS,
    ESI_SCORE_CRITICAL_MISS,
)


# ---------------------------------------------------------------------------
# score_esi
# ---------------------------------------------------------------------------

class TestScoreEsi:
    def test_perfect_match(self):
        score, undertriage = score_esi(1, 1)
        assert score == ESI_SCORE_PERFECT
        assert not undertriage

    def test_perfect_match_esi5(self):
        score, undertriage = score_esi(5, 5)
        assert score == ESI_SCORE_PERFECT
        assert not undertriage

    def test_off_by_one_non_critical(self):
        score, undertriage = score_esi(3, 4)
        assert score == ESI_SCORE_OFF_BY_ONE
        assert not undertriage

    def test_off_by_one_overtriage_on_critical(self):
        # ESI 1 overtriaged to ESI 2 — NOT undertriage, just off-by-one
        score, undertriage = score_esi(1, 2)
        assert score == ESI_SCORE_OFF_BY_ONE
        assert not undertriage

    def test_critical_miss_undertriage(self):
        # ESI 1 patient classified as ESI 2 — undertriage
        score, undertriage = score_esi(2, 1)
        assert score == ESI_SCORE_CRITICAL_MISS
        assert undertriage

    def test_critical_miss_esi2_classified_as_3(self):
        # ESI 2 patient classified as ESI 3 — undertriage
        score, undertriage = score_esi(3, 2)
        assert score == ESI_SCORE_CRITICAL_MISS
        assert undertriage

    def test_off_by_two(self):
        score, undertriage = score_esi(3, 1)
        assert score == ESI_SCORE_OFF_BY_TWO

    def test_large_miss(self):
        score, undertriage = score_esi(5, 1)
        assert score == ESI_SCORE_LARGE_MISS

    def test_none_inputs_return_zero(self):
        score, undertriage = score_esi(None, 1)
        assert score == 0.0
        assert not undertriage

    def test_symmetry_non_critical(self):
        # Off-by-1 in either direction on non-critical should be equal
        s1, _ = score_esi(4, 5)
        s2, _ = score_esi(5, 4)
        assert s1 == s2


# ---------------------------------------------------------------------------
# score_temporal
# ---------------------------------------------------------------------------

class TestScoreTemporal:
    def test_critical_no_extra_steps(self):
        # ESI 1, exactly minimum steps → full score
        score = score_temporal(esi_correct=1, steps_taken=2, expected_clarify_steps=1, clarify_count=1)
        assert score == 1.0

    def test_critical_one_extra_step(self):
        score = score_temporal(esi_correct=1, steps_taken=3, expected_clarify_steps=1, clarify_count=1)
        # 1.0 - 0.12 = 0.88
        assert abs(score - 0.88) < 1e-4

    def test_critical_many_extra_steps_clamped_to_zero(self):
        score = score_temporal(esi_correct=1, steps_taken=20, expected_clarify_steps=1, clarify_count=1)
        assert score == 0.0

    def test_noncritical_skip_penalty(self):
        # ESI 5, expected 1 clarify step, agent skipped
        score = score_temporal(esi_correct=5, steps_taken=1, expected_clarify_steps=1, clarify_count=0)
        assert abs(score - 0.95) < 1e-4

    def test_noncritical_no_penalty_if_clarified(self):
        score = score_temporal(esi_correct=5, steps_taken=2, expected_clarify_steps=1, clarify_count=1)
        assert score == 1.0

    def test_esi3_neutral(self):
        # ESI 3 has no temporal incentive either way
        score = score_temporal(esi_correct=3, steps_taken=10, expected_clarify_steps=5, clarify_count=0)
        assert score == 1.0


# ---------------------------------------------------------------------------
# score_reasoning — anti-gaming
# ---------------------------------------------------------------------------

class TestScoreReasoning:
    KEYWORDS = ["STEMI", "12-lead ECG", "troponin", "cath lab", "aspirin", "nitrate"]

    def test_full_coverage_verbose(self):
        reasoning = (
            "This presentation is consistent with STEMI given the chest pain and ECG changes. "
            "A 12-lead ECG confirms ST elevation. Troponin elevation supports myocardial injury. "
            "We need to activate the cath lab immediately. Aspirin and nitrate should be given. "
            "The patient requires urgent intervention to prevent further myocardial damage."
        )
        score = score_reasoning(reasoning, self.KEYWORDS)
        assert score >= 0.8, f"Expected high score for good reasoning, got {score}"

    def test_keyword_stuffing_penalised(self):
        # Keyword list repeated 6 times as bare tokens — should be penalised
        stuffed = " | ".join(self.KEYWORDS * 6)
        score = score_reasoning(stuffed, self.KEYWORDS)
        # Should score less than honest verbose reasoning
        honest_score = score_reasoning(
            "This is consistent with STEMI given the 12-lead ECG changes and troponin. "
            "Activate the cath lab, give aspirin and nitrate for this urgent presentation. "
            "Immediate reperfusion therapy is the priority.",
            self.KEYWORDS,
        )
        assert score < honest_score, (
            f"Stuffed score ({score}) should be lower than honest score ({honest_score})"
        )

    def test_empty_reasoning_zero(self):
        score = score_reasoning("", self.KEYWORDS)
        assert score == 0.0

    def test_no_keywords_long_reasoning_gets_partial_credit(self):
        score = score_reasoning(
            "The patient presents with a complex multi-system picture requiring careful assessment "
            "of the available clinical data before making a triage determination.",
            [],
        )
        assert score > 0.0

    def test_short_reasoning_capped(self):
        # Under 30 words, even with keywords, score should be capped
        short = "STEMI ECG troponin cath aspirin nitrate."
        score = score_reasoning(short, self.KEYWORDS)
        assert score <= 0.60


# ---------------------------------------------------------------------------
# keyword_matches — tier-2 now requires ALL tokens
# ---------------------------------------------------------------------------

class TestKeywordMatches:
    def test_exact_phrase_match(self):
        result = keyword_matches("activate cardiac cath lab now", ["cardiac cath lab"])
        assert result == ["cardiac cath lab"]

    def test_partial_single_token_should_not_match_multitoken_keyword(self):
        # Previously would match because "cath" is in text and is a sig token of "cath lab"
        # Now requires ALL significant tokens ("cath" AND "lab") to be present
        result = keyword_matches("activate cath immediately", ["cath lab"])
        # "lab" is missing, so this should NOT match
        assert result == [], f"Expected no match but got: {result}"

    def test_all_tokens_present_matches(self):
        result = keyword_matches("activate cath lab for stemi", ["cath lab"])
        assert result == ["cath lab"]

    def test_single_token_keyword_still_matches(self):
        result = keyword_matches("the ecg shows st elevation", ["ECG"])
        assert result == ["ECG"]

    def test_empty_text_returns_empty(self):
        result = keyword_matches("", ["stemi", "ecg"])
        assert result == []


# ---------------------------------------------------------------------------
# action_overlap — majority-token matching
# ---------------------------------------------------------------------------

class TestActionOverlap:
    def test_single_word_rec_should_not_match_long_expected(self):
        # "cath" alone should not match "activate cardiac catheterization laboratory"
        count, matched = action_overlap(
            ["cath"],
            ["activate cardiac catheterization laboratory"],
        )
        # "activate", "cardiac", "catheterization", "laboratory" — 4 significant tokens
        # "cath" appears in "catheterization" but NOT as a whole word token match
        # The test validates the majority threshold prevents false positives
        assert count == 0, f"Expected 0 matches but got {count}: {matched}"

    def test_majority_present_matches(self):
        # "cardiac catheterization laboratory" has 3 significant tokens: cardiac, catheterization, laboratory
        # "cardiac" and "catheterization" are 2/3 = ceil(3/2) = 2 → matches
        count, matched = action_overlap(
            ["activate the cardiac catheterization lab"],
            ["activate cardiac catheterization laboratory"],
        )
        assert count == 1

    def test_exact_match(self):
        count, matched = action_overlap(
            ["12-lead ECG immediately"],
            ["12-lead ECG"],
        )
        assert count == 1

    def test_empty_expected_returns_one(self):
        # score_actions returns 1.0 when expected is empty
        from triagerl.reward.components import score_actions
        score = score_actions(["anything"], [])
        assert score == 1.0

    def test_no_overlap(self):
        count, matched = action_overlap(
            ["reassure patient"],
            ["activate cardiac catheterization laboratory", "12-lead ECG", "IV access"],
        )
        assert count == 0


# ---------------------------------------------------------------------------
# score_actions integration
# ---------------------------------------------------------------------------

class TestScoreActions:
    def test_perfect_coverage(self):
        expected = ["12-lead ECG", "IV access and fluids", "cardiac monitor"]
        recommended = ["perform 12-lead ECG", "establish IV access and fluids", "attach cardiac monitor"]
        score = score_actions(recommended, expected)
        assert score == 1.0

    def test_zero_coverage(self):
        expected = ["activate cath lab", "aspirin administration", "nitrate sublingual"]
        recommended = ["reassure patient and observe"]
        score = score_actions(recommended, expected)
        assert score == 0.0

    def test_partial_coverage(self):
        expected = ["12-lead ECG", "IV access and fluids", "cardiac monitor", "aspirin"]
        recommended = ["12-lead ECG performed", "aspirin 300mg given"]
        score = score_actions(recommended, expected)
        assert abs(score - 0.5) < 0.01
