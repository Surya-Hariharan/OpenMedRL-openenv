"""
Shared constants for the Medical Triage RL Environment.

This module is the single source of truth for the keyword → trigger mapping
used by both the information-reveal logic (InfoRevealer) and the clinical-path
scoring (graders.py).  Keeping them synchronised here prevents the silent
scoring inconsistency where a question successfully unlocks hidden information
but does not earn the +0.30 path-quality bonus.

Maintenance rule
----------------
When you add a new trigger or keyword to InfoRevealer, add it HERE first.
Both InfoRevealer.KEYWORD_TO_TRIGGER and CLINICAL_TRIGGER_KEYWORDS in
graders.py are imported from this module and will stay in sync automatically.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Valid trigger values — must match what InfoRevealer.infer_trigger() returns
# ---------------------------------------------------------------------------

VALID_TRIGGERS: FrozenSet[str] = frozenset(
    {"ask_history", "check_vitals", "examine_patient"}
)

# ---------------------------------------------------------------------------
# Keyword → trigger vocabulary
# ---------------------------------------------------------------------------
# Keys are lowercase substrings that appear in natural-language clarifying
# questions.  Values are the trigger names used in HiddenInfoItem.trigger
# and returned by InfoRevealer.infer_trigger().
#
# Scoring rule: the trigger with the highest total matched-keyword-length
# wins (longer keywords are more specific and score higher).

KEYWORD_TO_TRIGGER: Dict[str, str] = {
    # ask_history — medical / social history questions
    "history":      "ask_history",
    "past":         "ask_history",
    "medication":   "ask_history",
    "medications":  "ask_history",
    "allergy":      "ask_history",
    "allergies":    "ask_history",
    "urine":        "ask_history",
    "urinary":      "ask_history",
    "uti":          "ask_history",
    "dysuria":      "ask_history",
    "frequency":    "ask_history",
    "burning":      "ask_history",

    # check_vitals — vital sign / monitoring questions
    "vital":         "check_vitals",
    "vitals":        "check_vitals",
    "bp":            "check_vitals",
    "blood pressure": "check_vitals",
    "heart rate":    "check_vitals",
    "spo2":          "check_vitals",
    "oxygen":        "check_vitals",
    "temperature":   "check_vitals",
    "gcs":           "check_vitals",

    # examine_patient — physical examination questions
    "examine":      "examine_patient",
    "exam":         "examine_patient",
    "physical":     "examine_patient",
    "inspection":   "examine_patient",
    "palpation":    "examine_patient",
    "listen":       "examine_patient",
    "auscultation": "examine_patient",
}
