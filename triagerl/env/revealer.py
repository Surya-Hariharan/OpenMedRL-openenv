"""
triagerl.env.revealer
=====================
Progressive information disclosure for the triage environment.

Responsibility
--------------
Given a free-text clarifying question from the agent, determine which
hidden information layer (if any) to unlock and return the revealed payload.

What this module does NOT do
-----------------------------
*  Does not apply vital drift — that is ``VitalDriftEngine`` (drift.py).
*  Does not compute rewards — that is ``graders.py``.
*  Does not mutate ``EpisodeState`` directly — the env orchestrator
   applies the returned payload via ``EpisodeState``.
*  Does not log at INFO level — only DEBUG so prod logs stay quiet.

Design notes
------------
*  ``InfoRevealer`` is constructed once per env instance and reset per
   episode.  Calling ``reset()`` without a seed lets the RNG advance freely
   for trajectory diversity — calling it with a seed replays deterministically.
*  The keyword-scoring algorithm is length-weighted: longer keywords carry
   more evidence weight than shorter synonyms.  ``"blood pressure"`` (13 chars)
   outweighs ``"bp"`` (2 chars).  This is clinically correct — specificity
   correlates with keyword length.
*  Direct trigger injection ("ask_history", "check_vitals", "examine_patient"
   appearing verbatim in the question) is blocked.  This prevents a trivially
   reward-hacking agent from bypassing natural-language keyword matching.
"""
from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any, Dict, FrozenSet, List, Optional, Set

from triagerl.core.constants import KEYWORD_TO_TRIGGER, VALID_TRIGGERS
from triagerl.tasks.schema import ImagingFinding, LabResult
from triagerl.tasks import TaskConfig


# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------

RevealPayload = Dict[str, Any]
"""
Structure returned by ``process_clarify()``.

Keys
----
trigger : str
    The trigger that was matched (member of VALID_TRIGGERS).
revealed : dict
    Raw key/value pairs from the hidden info layer(s).
labs : list[dict]
    Auto-extracted lab results (validated by caller into LabResult).
imaging : list[dict]
    Auto-extracted imaging findings (validated by caller into ImagingFinding).

An empty dict ``{}`` means no reveal occurred (low-quality question,
already-revealed trigger, or no matching hidden layer).
"""


# ---------------------------------------------------------------------------
# Token sets for auto-extraction
# ---------------------------------------------------------------------------

_IMAGING_KEY_TOKENS: FrozenSet[str] = frozenset({
    "ecg", "ct", "xray", "x-ray", "mri", "ultrasound", "echo", "cxr",
})

_LAB_KEY_TOKENS: FrozenSet[str] = frozenset({
    "inr", "lactate", "troponin", "bnp", "wcc", "crp", "creatinine",
    "anti-xa", "glucose",
})

_LAB_CRITICAL_FLAGS: FrozenSet[str] = frozenset({
    "critical", "high", "low", "supratherapeutic",
})

# Minimum word count for a clarifying question to pass the quality gate.
_MIN_WORDS: int = 4

# Contextual words that indicate intent even without a "?" character.
_INTENT_WORDS: FrozenSet[str] = frozenset({
    "what", "when", "how", "any", "history", "exam", "vitals", "changes",
})

# BP parsing — "check_vitals" reveals often contain textual BP strings.
_BP_KEYS_IN_ORDER = ("repeat_bp", "bp", "blood_pressure")


# ---------------------------------------------------------------------------
# InfoRevealer
# ---------------------------------------------------------------------------

class InfoRevealer:
    """
    Manages progressive hidden-information disclosure for one task.

    One instance is created per ``MedicalTriageEnv`` and persists across
    episodes.  ``reset()`` clears reveal state between episodes.

    Parameters
    ----------
    task : TaskConfig
        The task whose ``hidden_info`` list defines what can be revealed.
    rng : random.Random | None
        Optional pre-seeded RNG.  If ``None``, one is created from ``seed``.
    seed : int | None
        Seed passed to ``random.Random`` if no ``rng`` is provided.
        The same seed produces the same drift sequence — useful for replay.
    """

    # Class-level exposure of the canonical maps so callers can introspect
    # without importing constants.py directly.
    KEYWORD_TO_TRIGGER: Dict[str, str]       = KEYWORD_TO_TRIGGER
    DIRECT_TRIGGER_TOKENS: FrozenSet[str]    = VALID_TRIGGERS

    def __init__(
        self,
        task: TaskConfig,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._task: TaskConfig           = task
        self._seed: Optional[int]        = seed
        self._rng:  random.Random        = rng or random.Random(seed)
        self._revealed_triggers: Set[str] = set()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Clear per-episode reveal state.

        When ``seed`` is ``None`` (default) the RNG is NOT reseeded — it
        continues from its current position, producing diverse drift
        sequences across successive episodes on the same object.

        When ``seed`` is an integer the RNG is reseeded to that value for
        deterministic replay of a specific episode.

        Do **not** pass the constructor seed on every call if you want
        trajectory diversity — each call would replay the same sequence.
        """
        self._revealed_triggers.clear()
        if seed is not None:
            self._seed = seed
            self._rng.seed(seed)

    # ------------------------------------------------------------------
    # Initial vitals (called once per reset)
    # ------------------------------------------------------------------

    def get_initial_vitals(self) -> Dict[str, Any]:
        """
        Return the visible subset of initial vitals, masking hidden keys.

        Keys listed under ``"hidden_vitals"`` in any ``HiddenInfoItem.data``
        are removed from the returned dict.  They will only become visible
        after the appropriate trigger fires.
        """
        vitals = deepcopy(self._task.initial_vitals)

        hidden_keys: Set[str] = set()
        for item in self._task.hidden_info:
            if "hidden_vitals" in item.data:
                hidden_keys.update(item.data["hidden_vitals"])

        for key in hidden_keys:
            vitals.pop(key, None)

        return vitals

    # ------------------------------------------------------------------
    # Clarify processing
    # ------------------------------------------------------------------

    def process_clarify(self, question: str, step: int) -> RevealPayload:
        """
        Evaluate a clarifying question and return the reveal payload.

        Returns an empty dict ``{}`` when:
          * The question is too short or is a direct trigger injection (quality gate).
          * The inferred trigger has already been revealed this episode.
          * No ``HiddenInfoItem`` in the task matches the inferred trigger.

        Otherwise returns a ``RevealPayload`` dict (see type alias above).

        Parameters
        ----------
        question : str
            The agent's free-text clarifying question.
        step : int
            Current episode step (used for logging only).
        """
        if not self._is_meaningful(question):
            return {}

        trigger = self._infer_trigger(question)

        # "clarify" sentinel means no keyword matched — nothing to unlock.
        if trigger == "clarify":
            return {}

        # Already revealed this trigger in the current episode — idempotent no-op.
        if trigger in self._revealed_triggers:
            return {}

        # Find all hidden info layers that match this trigger.
        matching_layers = [
            item.data
            for item in self._task.hidden_info
            if item.trigger == trigger
        ]

        if not matching_layers:
            # Trigger is valid but task has no data behind it.
            return {}

        # Register as revealed BEFORE building payload so re-entry is safe
        # even if payload construction throws.
        self._revealed_triggers.add(trigger)

        payload = self._build_payload(trigger, matching_layers)
        return payload

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def revealed_trigger_keys(self) -> List[str]:
        """Ordered list of triggers that have been successfully revealed."""
        return sorted(self._revealed_triggers)

    def is_trigger_revealed(self, trigger: str) -> bool:
        """True if ``trigger`` has already been unlocked this episode."""
        return trigger in self._revealed_triggers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_meaningful(self, question: str) -> bool:
        """
        Quality gate: reject questions that are too short or are direct
        trigger injections.

        A question passes if:
          1. It has at least ``_MIN_WORDS`` whitespace-separated tokens.
          2. It does not contain any verbatim DIRECT_TRIGGER_TOKEN string.
          3. It contains either "?" OR at least one ``_INTENT_WORD``.

        Condition 3 is intentionally permissive — any clinical question
        containing "any", "history", "vitals", etc. passes even without
        a question mark, so well-formed statements also qualify.
        """
        text = (question or "").strip().lower()

        # Length gate
        if len(text.split()) < _MIN_WORDS:
            return False

        # Direct injection gate (reward hacking prevention)
        if any(token in text for token in self.DIRECT_TRIGGER_TOKENS):
            return False

        # Intent gate
        if "?" in text:
            return True
        return any(word in text for word in _INTENT_WORDS)

    def _infer_trigger(self, question: str) -> str:
        """
        Score the question against the keyword vocabulary and return the
        best-matching trigger name.

        Algorithm
        ---------
        For each keyword that appears as a substring in the lowercased
        question, add ``len(keyword)`` to that keyword's trigger score.
        The trigger with the highest cumulative score wins.  Ties are broken
        alphabetically for determinism.

        Returns ``"clarify"`` (the sentinel) when no keyword matches.
        """
        text = question.lower()

        # Defensive: block direct injection before scoring.
        if any(token in text for token in self.DIRECT_TRIGGER_TOKENS):
            return "clarify"

        scores: Dict[str, int] = {}
        for keyword, trigger in self.KEYWORD_TO_TRIGGER.items():
            if keyword in text:
                scores[trigger] = scores.get(trigger, 0) + len(keyword)

        if not scores:
            return "clarify"

        # max over sorted keys for tie-breaking determinism.
        return max(sorted(scores), key=lambda t: scores[t])

    def _build_payload(
        self,
        trigger: str,
        layers: List[Dict[str, Any]],
    ) -> RevealPayload:
        """
        Merge all matching data layers into a single ``RevealPayload``.

        Auto-extraction rules
        ---------------------
        *  Keys containing imaging tokens (ecg, ct, cxr, …) are extracted
           into ``imaging`` as ``ImagingFinding``-compatible dicts.
        *  Keys containing lab tokens (inr, troponin, bnp, …) are extracted
           into ``labs`` as ``LabResult``-compatible dicts.
        *  BP text strings in keys ``repeat_bp``, ``bp``, ``blood_pressure``,
           or ``bp_differential`` are parsed into structured systolic/diastolic
           pairs and returned under the ``"vitals_update"`` key for the env
           to apply.

        The raw ``revealed`` dict always contains the full merged payload
        regardless of auto-extraction — callers may derive additional
        structure from it independently.
        """
        revealed: Dict[str, Any] = {}
        labs:     List[Dict[str, Any]] = []
        imaging:  List[Dict[str, Any]] = []

        for layer in layers:
            for k, v in layer.items():
                revealed[k] = v
                key_low  = k.lower()
                text_val = str(v)

                if any(tok in key_low for tok in _IMAGING_KEY_TOKENS):
                    imaging.append({
                        "modality": "clinical finding",
                        "finding":  f"{k}: {text_val}",
                        "critical": False,
                    })

                if any(tok in key_low for tok in _LAB_KEY_TOKENS):
                    labs.append({
                        "name":            k,
                        "value":           text_val,
                        "unit":            "",
                        "reference_range": "",
                        "critical":        any(
                            flag in text_val.lower()
                            for flag in _LAB_CRITICAL_FLAGS
                        ),
                    })

        vitals_update = self._extract_bp_update(revealed)

        payload: RevealPayload = {
            "trigger":        trigger,
            "revealed":       revealed,
            "labs":           labs,
            "imaging":        imaging,
        }
        if vitals_update:
            payload["vitals_update"] = vitals_update

        return payload

    @staticmethod
    def _extract_bp_update(revealed: Dict[str, Any]) -> Dict[str, int]:
        """
        Parse structured BP values from common textual patterns.

        For differential BP strings (e.g. aortic dissection:
        "Right arm 148/92 — Left arm 96/58"), the clinically significant
        reading is the LOWER systolic — it indicates obstruction or
        dissection, not the higher "normal" arm.

        Returns an empty dict if no parseable BP string is found.
        """
        # Try direct vitals dict first (already structured).
        if isinstance(revealed.get("vitals"), dict):
            return {}  # caller will use revealed["vitals"] directly

        # Try named BP text keys.
        bp_text: Optional[str] = None
        for key in _BP_KEYS_IN_ORDER:
            if isinstance(revealed.get(key), str):
                bp_text = revealed[key]
                break

        if bp_text is None and isinstance(revealed.get("bp_differential"), str):
            bp_text = revealed["bp_differential"]

        if not bp_text:
            return {}

        pairs = re.findall(r"(\d{2,3})\s*/\s*(\d{2,3})", bp_text)
        if not pairs:
            return {}

        try:
            sys_val = min(int(s) for s, _ in pairs)
            dia_val = next(int(d) for s, d in pairs if int(s) == sys_val)
            return {
                "blood_pressure_systolic":  sys_val,
                "blood_pressure_diastolic": dia_val,
            }
        except (StopIteration, ValueError):
            return {}