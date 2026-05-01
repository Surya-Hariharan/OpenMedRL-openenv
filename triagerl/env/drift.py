"""
triagerl.env.drift
==================
Stochastic vital sign deterioration engine and deterioration signal computation.

Responsibility
--------------
*  ``VitalDriftEngine`` applies per-step Gaussian-noised drift to a mutable
   vitals dict.  It owns the RNG and all per-vital clamping rules.
*  ``compute_deterioration_signal`` is a pure function that estimates patient
   deterioration from visible vitals only — it does NOT consume hidden data
   and is NOT a reward signal.

What this module does NOT do
-----------------------------
*  Does not read or write ``EpisodeState`` — receives and returns dicts.
*  Does not compute rewards.
*  Does not log at INFO level.
*  Does not validate Pydantic models — callers are responsible for running
   ``VitalSigns.model_validate()`` on the returned dict before storing it.

Design notes
------------
*  Drift is deliberately Gaussian-noised rather than deterministic so that
   each RL rollout with the same task produces a different vital trajectory.
   This prevents the training agent from memorising the exact step at which
   a threshold is crossed.
*  Clamping ranges are physiologically motivated hard limits, not soft
   guidance.  Values outside these ranges are clinically impossible and
   would distort the deterioration signal.
*  ``compute_deterioration_signal`` is designed to be safe for the agent to
   see — it is computed from *visible* vitals only, so it cannot function as
   a cheat code exposing hidden information.
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, Optional

from triagerl.tasks import VitalDrift


# ---------------------------------------------------------------------------
# Per-vital physiological clamp bounds
# ---------------------------------------------------------------------------

_CLAMP_BOUNDS: Dict[str, tuple[float, float]] = {
    "heart_rate":              (20.0,  300.0),
    "oxygen_saturation":       (50.0,  100.0),
    "blood_pressure_systolic": (40.0,  300.0),
    "respiratory_rate":        (4.0,   60.0),
    "gcs":                     (3.0,   15.0),
    # temperature and diastolic BP are unclamped — physiological extremes
    # for these are task-dependent and handled by VitalSigns validators.
}

# Vitals that should be stored as floats (1 decimal place) rather than ints.
_FLOAT_VITALS: frozenset[str] = frozenset({"temperature", "oxygen_saturation"})


# ---------------------------------------------------------------------------
# VitalDriftEngine
# ---------------------------------------------------------------------------

class VitalDriftEngine:
    """
    Applies stochastic per-step vital sign deterioration.

    Constructed once per ``MedicalTriageEnv`` with the task's ``VitalDrift``
    configuration.  The same engine instance is reused across episode resets
    — call ``reset()`` with a seed to replay deterministically or without
    a seed for diverse trajectories.

    Parameters
    ----------
    drift_config : VitalDrift
        Frozen Pydantic model from ``TaskConfig.vital_drift``.  Contains:
          - ``per_step`` : dict mapping vital name → mean drift per step.
          - ``noise_sigma`` : dict mapping vital name → Gaussian noise sigma.
          - ``starts_at_step`` : drift activation step (0-indexed).
    rng : random.Random | None
        Pre-seeded RNG to share with ``InfoRevealer`` if desired.
        Passing ``None`` creates an independent RNG from ``seed``.
    seed : int | None
        Seed for the internal RNG.  Ignored if ``rng`` is provided.
    """

    def __init__(
        self,
        drift_config: VitalDrift,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config: VitalDrift    = drift_config
        self._seed:   Optional[int] = seed
        self._rng:    random.Random = rng or random.Random(seed)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset drift RNG state.

        When ``seed`` is ``None`` (default) the RNG continues from its
        current state — successive episodes produce distinct trajectories.
        When ``seed`` is an integer the RNG is reseeded for deterministic
        replay.
        """
        if seed is not None:
            self._seed = seed
            self._rng.seed(seed)

    # ------------------------------------------------------------------
    # Core drift application
    # ------------------------------------------------------------------

    def apply(self, vitals: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Apply one step of stochastic drift and return the updated vitals.

        Returns the original dict (deep-copied) unchanged if:
          * ``step`` is before the configured ``starts_at_step``.
          * ``per_step`` is empty (no drift configured).

        Parameters
        ----------
        vitals : dict
            Current vital-sign snapshot.  Keys must match VitalSigns field
            names.  None-valued keys are skipped silently.
        step : int
            The current episode step (1-indexed from env perspective).

        Returns
        -------
        dict
            A deep copy of ``vitals`` with drift applied.  The caller is
            responsible for running ``VitalSigns.model_validate()`` on the
            result before storing.
        """
        if step < self._config.starts_at_step:
            return vitals

        if not self._config.per_step:
            return vitals

        drifted = deepcopy(vitals)

        for vital_key, mean_drift in self._config.per_step.items():
            current = drifted.get(vital_key)
            if current is None:
                continue

            current_f = float(current)
            sigma = float(self._config.noise_sigma.get(vital_key, 0.0))
            noise = self._rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0
            new_val = current_f + mean_drift + noise

            # Physiological clamping — only applied to keys with defined bounds.
            if vital_key in _CLAMP_BOUNDS:
                lo, hi = _CLAMP_BOUNDS[vital_key]
                new_val = max(lo, min(hi, new_val))

            # Type coercion — keep floats as floats, ints as ints.
            if vital_key in _FLOAT_VITALS:
                drifted[vital_key] = round(new_val, 1)
            else:
                drifted[vital_key] = int(round(new_val))

        return drifted


# ---------------------------------------------------------------------------
# Deterioration signal (pure function — no state)
# ---------------------------------------------------------------------------

def compute_deterioration_signal(
    current_vitals: Dict[str, Any],
    baseline_vitals: Dict[str, Any],
) -> float:
    """
    Estimate patient deterioration from publicly visible vitals only.

    Returns a float in [0.0, 1.0].  Higher values indicate greater
    deterioration from the initial observation.

    This is NOT a reward signal — it is an agent-facing urgency hint
    included in ``TriageObservation.deterioration_signal``.

    Algorithm
    ---------
    For each vital present in both ``current`` and ``baseline``:

      * ``oxygen_saturation``  : deterioration ∝ drop / 20 (10-pt drop ≈ 0.5)
      * ``blood_pressure_systolic`` : deterioration ∝ drop / 40
      * ``gcs``               : deterioration ∝ drop / 4
      * ``heart_rate``, ``respiratory_rate``, ``temperature`` :
            deterioration ∝ abs(Δ) / max(10, 25% of baseline)

    The per-vital deltas are averaged and clamped to [0, 1].

    Parameters
    ----------
    current_vitals : dict
        Current vital-sign snapshot (already clamped by VitalSigns validator).
    baseline_vitals : dict
        The initial visible vitals captured at reset time.

    Returns
    -------
    float
        Deterioration signal ∈ [0.0, 1.0], rounded to 4 decimal places.
    """
    if not baseline_vitals:
        return 0.0

    score = 0.0
    count = 0

    for key, baseline in baseline_vitals.items():
        if baseline is None:
            continue
        current = current_vitals.get(key)
        if current is None:
            continue

        count += 1
        now  = float(current)
        base = float(baseline)

        if key == "oxygen_saturation":
            delta = max(0.0, (base - now) / 20.0)
        elif key == "blood_pressure_systolic":
            delta = max(0.0, (base - now) / 40.0)
        elif key == "gcs":
            delta = max(0.0, (base - now) / 4.0)
        elif key in {"heart_rate", "respiratory_rate", "temperature"}:
            delta = abs(now - base) / max(10.0, abs(base) * 0.25)
        else:
            delta = 0.0

        score += min(1.0, delta)

    if count == 0:
        return 0.0

    return round(max(0.0, min(1.0, score / count)), 4)


# ---------------------------------------------------------------------------
# Time-pressure note (pure function — no state)
# ---------------------------------------------------------------------------

_HIGH_DETERIORATION_THRESHOLD:  float = 0.6
_WARN_DETERIORATION_THRESHOLD:  float = 0.4
_CRITICAL_STEPS_REMAINING:      int   = 2
_URGENT_STEPS_REMAINING:        int   = 1


def compute_time_pressure_note(
    deterioration_signal: float,
    steps_remaining: int,
    is_done: bool,
) -> Optional[str]:
    """
    Return a human-readable urgency note for ``TriageObservation``.

    Returns ``None`` when no special urgency applies.

    Parameters
    ----------
    deterioration_signal : float
        Output of ``compute_deterioration_signal()``.
    steps_remaining : int
        ``max_steps - current_step`` computed by the env orchestrator.
    is_done : bool
        True when the episode has already terminated — suppress notes.
    """
    if is_done:
        return None

    if deterioration_signal >= _HIGH_DETERIORATION_THRESHOLD and steps_remaining <= _CRITICAL_STEPS_REMAINING:
        return "Patient deteriorating rapidly — disposition should not be delayed."

    if deterioration_signal >= _WARN_DETERIORATION_THRESHOLD:
        return "Clinical trajectory worsening — prioritize critical interventions and triage decision."

    if steps_remaining <= _URGENT_STEPS_REMAINING:
        return "Final step remaining — complete triage disposition now."

    return None