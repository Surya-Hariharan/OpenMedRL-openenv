"""
triagerl.core.constants
=======================
Single source of truth for every keyword → trigger mapping and trigger
vocabulary used across the entire system.

Maintenance contract
--------------------
This is the ONLY place where keyword/trigger data lives.

The following modules MUST import from here rather than define their own copy:

    - env/revealer.py      (InfoRevealer.KEYWORD_TO_TRIGGER)
    - reward/path_quality.py (clinical path scoring)
    - eval/metrics.py      (useful-clarify counting in EpisodeMetrics)

Adding a keyword
----------------
1.  Add the entry to KEYWORD_TO_TRIGGER below.
2.  If the target trigger is new, add it to VALID_TRIGGERS.
3.  Run the unit tests in tests/core/test_constants.py to verify coverage.
4.  No other file needs to change.

Design notes
------------
*  Longer keywords score higher in trigger disambiguation because
   `infer_trigger()` sums matched keyword *lengths* rather than counts.
   This means "blood pressure" (13 chars) outweighs "bp" (2 chars) when
   both appear in the same question — which is clinically correct behaviour.
*  The three trigger names are deliberately human-readable so that task
   YAML authors can reference them without consulting code.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Valid trigger names
# ---------------------------------------------------------------------------
# These are the only values that HiddenInfoItem.trigger may hold and the only
# values that InfoRevealer.infer_trigger() may return (aside from the sentinel
# "clarify" returned when no keyword matches).
#
# "clarify" is intentionally excluded here — it is a sentinel, not a
# named trigger.  Using "clarify" as a HiddenInfoItem.trigger would cause
# any low-quality question that matches no keyword to unlock that layer,
# which is a reward-hacking vector.

VALID_TRIGGERS: FrozenSet[str] = frozenset({
    "ask_history",
    "check_vitals",
    "examine_patient",
})

# ---------------------------------------------------------------------------
# Keyword → trigger vocabulary
# ---------------------------------------------------------------------------
# Keys   : lowercase substrings searched in the agent's clarifying question.
# Values : trigger name (must be a member of VALID_TRIGGERS).
#
# Disambiguation rule (implemented in InfoRevealer.infer_trigger):
#   For each keyword that appears in the question, add len(keyword) to that
#   trigger's score.  The trigger with the highest total score wins.
#   Ties are broken alphabetically (deterministic, avoids dict-ordering bugs).
#
# Grouping convention:
#   Keep keywords for the same trigger together and ordered longest → shortest
#   so the file is easy to scan and accidental duplicate entries are obvious.

KEYWORD_TO_TRIGGER: Dict[str, str] = {
    # ── ask_history ──────────────────────────────────────────────────────────
    # Questions about the patient's past, symptoms, medications, or urinary
    # tract complaints unlock the ask_history reveal layer.
    "medical history":    "ask_history",    # 15 chars — highest specificity
    "past history":       "ask_history",    # 12 chars
    "medications":        "ask_history",    # 11 chars
    "medication":         "ask_history",    # 10 chars
    "allergies":          "ask_history",    #  8 chars
    "dysuria":            "ask_history",    #  7 chars
    "urinary":            "ask_history",    #  7 chars
    "burning":            "ask_history",    #  7 chars
    "history":            "ask_history",    #  7 chars
    "allergy":            "ask_history",    #  6 chars
    "urine":              "ask_history",    #  5 chars
    "past":               "ask_history",    #  4 chars
    "uti":                "ask_history",    #  3 chars

    # ── check_vitals ─────────────────────────────────────────────────────────
    # Questions about measured physiological parameters unlock the
    # check_vitals reveal layer.
    "blood pressure":     "check_vitals",   # 14 chars — highest specificity
    "oxygen saturation":  "check_vitals",   # 17 chars
    "respiratory rate":   "check_vitals",   # 15 chars
    "heart rate":         "check_vitals",   # 10 chars
    "temperature":        "check_vitals",   # 11 chars
    "saturation":         "check_vitals",   # 10 chars
    "glasgow":            "check_vitals",   #  7 chars
    "vitals":             "check_vitals",   #  6 chars
    "vital":              "check_vitals",   #  5 chars
    "spo2":               "check_vitals",   #  4 chars
    "gcs":                "check_vitals",   #  3 chars
    "bp":                 "check_vitals",   #  2 chars — lowest specificity

    # ── examine_patient ───────────────────────────────────────────────────────
    # Questions requesting a physical examination unlock the
    # examine_patient reveal layer.
    "auscultation":       "examine_patient",  # 12 chars — highest specificity
    "percussion":         "examine_patient",  # 10 chars
    "inspection":         "examine_patient",  # 10 chars
    "palpation":          "examine_patient",  #  9 chars
    "examine":            "examine_patient",  #  7 chars
    "physical":           "examine_patient",  #  8 chars
    "listen":             "examine_patient",  #  6 chars
    "exam":               "examine_patient",  #  4 chars
}

# ---------------------------------------------------------------------------
# Derived lookup: trigger → all keywords that map to it
# ---------------------------------------------------------------------------
# Pre-computed so callers that need reverse lookup don't re-derive it.
# This dict is immutable by convention — do not mutate at runtime.

TRIGGER_TO_KEYWORDS: Dict[str, list[str]] = {}
for _kw, _trigger in KEYWORD_TO_TRIGGER.items():
    TRIGGER_TO_KEYWORDS.setdefault(_trigger, []).append(_kw)
# Sort each keyword list longest → shortest (matches disambiguation priority)
for _trigger in TRIGGER_TO_KEYWORDS:
    TRIGGER_TO_KEYWORDS[_trigger].sort(key=len, reverse=True)

# ---------------------------------------------------------------------------
# Direct-trigger injection tokens (used by InfoRevealer as a blocklist)
# ---------------------------------------------------------------------------
# If an agent includes any of these strings verbatim in a clarifying question,
# the question is rejected as a reward-hacking attempt.  The set equals
# VALID_TRIGGERS — any named trigger token injected directly is blocked.

DIRECT_TRIGGER_TOKENS: FrozenSet[str] = VALID_TRIGGERS

# ---------------------------------------------------------------------------
# Minimum meaningful clarification length (words)
# ---------------------------------------------------------------------------
# A clarifying question shorter than this is considered low-quality and will
# not unlock any hidden information layer.  Calibrated so that four-word
# clinical questions ("Any relevant past history?") pass the gate while
# single-word injections ("vitals") are rejected.

MIN_CLARIFICATION_WORDS: int = 4

# ---------------------------------------------------------------------------
# Integrity check (runs at import time, ~zero cost)
# ---------------------------------------------------------------------------
# Ensures no keyword maps to an undeclared trigger.  Catches typos introduced
# during keyword additions before they silently produce no-op reveals.

_bad = {kw: t for kw, t in KEYWORD_TO_TRIGGER.items() if t not in VALID_TRIGGERS}
if _bad:
    raise ValueError(
        f"KEYWORD_TO_TRIGGER contains keywords mapped to invalid triggers: {_bad}. "
        f"Valid triggers are: {sorted(VALID_TRIGGERS)}"
    )
del _bad