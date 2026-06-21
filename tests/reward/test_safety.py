"""
Tests for triagerl.reward.safety

Covers: undertriage detection, sign-aware modifier, boundary cases.
"""
import pytest

from triagerl.reward.safety import (
    apply_safety_modifier,
    is_undertriage,
    UNDERTRIAGE_MULTIPLIER,
    CRITICAL_ESI_THRESHOLD,
)


class TestIsUndertriage:
    def test_critical_patient_classified_higher(self):
        assert is_undertriage(correct_esi=1, predicted_esi=2) is True
        assert is_undertriage(correct_esi=1, predicted_esi=3) is True
        assert is_undertriage(correct_esi=2, predicted_esi=3) is True

    def test_exact_correct_is_not_undertriage(self):
        assert is_undertriage(correct_esi=1, predicted_esi=1) is False
        assert is_undertriage(correct_esi=2, predicted_esi=2) is False

    def test_non_critical_patient_is_not_undertriage(self):
        # ESI 3+ patients: higher predicted ESI is not undertriage in clinical terms
        assert is_undertriage(correct_esi=3, predicted_esi=4) is False
        assert is_undertriage(correct_esi=4, predicted_esi=5) is False

    def test_overtriage_is_not_undertriage(self):
        # Predicted lower than correct = overtriage, not undertriage
        assert is_undertriage(correct_esi=3, predicted_esi=1) is False


class TestApplySafetyModifier:
    def test_no_undertriage_returns_raw(self):
        adjusted, factor = apply_safety_modifier(0.75, undertriage=False, multiplier=1.0)
        assert abs(adjusted - 0.75) < 1e-6
        assert factor == 1.0

    def test_undertriage_reduces_positive_score(self):
        adjusted, factor = apply_safety_modifier(0.80, undertriage=True, multiplier=UNDERTRIAGE_MULTIPLIER)
        assert adjusted < 0.80
        assert factor == UNDERTRIAGE_MULTIPLIER
        assert abs(adjusted - 0.80 * UNDERTRIAGE_MULTIPLIER) < 1e-6

    def test_undertriage_magnifies_negative_score(self):
        # Sign-aware: negative raw should become more negative under undertriage
        raw = -0.20
        adjusted, factor = apply_safety_modifier(raw, undertriage=True, multiplier=UNDERTRIAGE_MULTIPLIER)
        assert adjusted < raw, f"Expected {adjusted} < {raw} (more negative)"

    def test_zero_raw_under_undertriage(self):
        # Edge case from bug fix: raw=0.0 should not crash and should return 0
        adjusted, factor = apply_safety_modifier(0.0, undertriage=True, multiplier=UNDERTRIAGE_MULTIPLIER)
        assert adjusted == 0.0

    def test_undertriage_multiplier_value(self):
        # The multiplier must be significantly less than 1.0 for safety to matter
        assert UNDERTRIAGE_MULTIPLIER <= 0.30, (
            f"UNDERTRIAGE_MULTIPLIER={UNDERTRIAGE_MULTIPLIER} is too lenient — "
            "should be ≤ 0.30 for meaningful safety signal"
        )

    def test_critical_threshold_is_2(self):
        # ESI ≤ 2 are critical — validate the constant
        assert CRITICAL_ESI_THRESHOLD == 2
