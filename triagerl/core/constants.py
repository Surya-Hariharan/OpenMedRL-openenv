"""
triagerl.core.constants
=======================
Single source of truth for all trigger and keyword constants used across the
TriageRL system.

Rules
-----
*  NO imports from any other triagerl module — this module sits at the
   absolute bottom of the dependency graph.  Any other module may import
   from here; nothing here imports from anywhere else in the project.
*  Every module that needs KEYWORD_TO_TRIGGER or VALID_TRIGGERS MUST import
   from here.  Local copies are forbidden.
*  Adding a new trigger requires three changes in this file only:
     1. Add the trigger name string to VALID_TRIGGERS.
     2. Add all keywords that should map to it in KEYWORD_TO_TRIGGER.
     3. Update the docstring below.

Trigger definitions
-------------------
ask_history
    Questions about medical history, medications, symptom timeline,
    family history, social history, allergies, or prior investigations.

check_vitals
    Requests for vital signs: blood pressure, heart rate, respiratory rate,
    oxygen saturation, temperature, GCS, or repeat/trended measurements.

examine_patient
    Requests for physical examination findings: inspection, palpation,
    auscultation, percussion, or targeted system-specific findings.

KEYWORD_TO_TRIGGER algorithm (used by InfoRevealer and path_quality)
--------------------------------------------------------------------
For each keyword present in a lowercased question string, the corresponding
trigger accumulates ``len(keyword)`` score points.  The trigger with the
highest cumulative score wins.  Ties are broken alphabetically by trigger
name.  Longer keywords are worth more points, which naturally gives
precedence to more specific clinical terms over generic ones.

This algorithm is authoritative — both ``InfoRevealer.infer_trigger()`` and
``score_clinical_path()`` use it via this shared mapping.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Valid trigger names
# ---------------------------------------------------------------------------

VALID_TRIGGERS: FrozenSet[str] = frozenset({
    "ask_history",
    "check_vitals",
    "examine_patient",
})
"""
The complete set of legal trigger strings.

Used by:
  - ``triagerl.tasks.schema.HiddenInfoItem.validate_trigger``
  - ``triagerl.tasks.corpus.tasks.yaml`` validation rules
  - ``triagerl.reward.path_quality`` (via KEYWORD_TO_TRIGGER)
  - ``triagerl.reward.shaping`` (_DIRECT_TRIGGER_TOKENS)
"""

# ---------------------------------------------------------------------------
# Keyword → trigger mapping
# ---------------------------------------------------------------------------

KEYWORD_TO_TRIGGER: Dict[str, str] = {
    # ── ask_history ───────────────────────────────────────────────────────────
    # Symptom history and timeline
    "history":                  "ask_history",
    "past history":             "ask_history",
    "medical history":          "ask_history",
    "past medical":             "ask_history",
    "previous history":         "ask_history",
    "symptom":                  "ask_history",
    "symptoms":                 "ask_history",
    "onset":                    "ask_history",
    "duration":                 "ask_history",
    "progression":              "ask_history",
    "timeline":                 "ask_history",
    "when did":                 "ask_history",
    "how long":                 "ask_history",
    "started":                  "ask_history",
    "worse":                    "ask_history",
    "better":                   "ask_history",
    "character":                "ask_history",
    "severity":                 "ask_history",
    "radiation":                "ask_history",
    "aggravating":              "ask_history",
    "relieving":                "ask_history",
    # Medications and adherence
    "medication":               "ask_history",
    "medications":              "ask_history",
    "medicine":                 "ask_history",
    "medicines":                "ask_history",
    "drug":                     "ask_history",
    "drugs":                    "ask_history",
    "prescription":             "ask_history",
    "anticoagulant":            "ask_history",
    "anticoagulants":           "ask_history",
    "warfarin":                 "ask_history",
    "apixaban":                 "ask_history",
    "heparin":                  "ask_history",
    "aspirin":                  "ask_history",
    "insulin":                  "ask_history",
    "beta blocker":             "ask_history",
    "bisoprolol":               "ask_history",
    "adherence":                "ask_history",
    "compliance":               "ask_history",
    "dose":                     "ask_history",
    "last dose":                "ask_history",
    "sick day":                 "ask_history",
    # Allergies
    "allergy":                  "ask_history",
    "allergies":                "ask_history",
    "allergic":                 "ask_history",
    "reaction":                 "ask_history",
    "anaphylaxis":              "ask_history",
    # Social and family history
    "social history":           "ask_history",
    "family history":           "ask_history",
    "smoking":                  "ask_history",
    "alcohol":                  "ask_history",
    "drug use":                 "ask_history",
    "occupation":               "ask_history",
    "travel":                   "ask_history",
    "contact":                  "ask_history",
    "sexual history":           "ask_history",
    # Past investigations and procedures
    "previous ecg":             "ask_history",
    "prior ecg":                "ask_history",
    "previous echo":            "ask_history",
    "angiogram":                "ask_history",
    "inr":                      "ask_history",
    "last inr":                 "ask_history",
    "last blood":               "ask_history",
    "blood test":               "ask_history",
    "investigation":            "ask_history",
    # Fluid and oral intake
    "fluid":                    "ask_history",
    "fluid intake":             "ask_history",
    "oral intake":              "ask_history",
    "eating":                   "ask_history",
    "drinking":                 "ask_history",
    "urine":                    "ask_history",
    "urine output":             "ask_history",
    "urine appearance":         "ask_history",
    "urine colour":             "ask_history",
    "urinary":                  "ask_history",
    "dysuria":                  "ask_history",
    # Weight
    "weight":                   "ask_history",
    "weight gain":              "ask_history",
    "weight loss":              "ask_history",
    # Pregnancy and obstetric
    "pregnancy":                "ask_history",
    "pregnant":                 "ask_history",
    "lmp":                      "ask_history",
    "last menstrual":           "ask_history",
    "obstetric":                "ask_history",
    "bhcg":                     "ask_history",
    "beta hcg":                 "ask_history",
    # Additional context
    "baseline":                 "ask_history",
    "functional baseline":      "ask_history",
    "cognitive baseline":       "ask_history",
    "home observation":         "ask_history",
    "carer":                    "ask_history",
    "family report":            "ask_history",
    "sick day rules":           "ask_history",
    "anticoagulation":          "ask_history",
    "anticoagulation status":   "ask_history",
    "known":                    "ask_history",

    # ── check_vitals ──────────────────────────────────────────────────────────
    # Vital sign terms
    "vital":                    "check_vitals",
    "vitals":                   "check_vitals",
    "vital signs":              "check_vitals",
    "vital sign":               "check_vitals",
    "observations":             "check_vitals",
    "obs":                      "check_vitals",
    # Blood pressure
    "blood pressure":           "check_vitals",
    "bp":                       "check_vitals",
    "systolic":                 "check_vitals",
    "diastolic":                "check_vitals",
    "bilateral bp":             "check_vitals",
    "bp differential":          "check_vitals",
    "both arms":                "check_vitals",
    "mean arterial":            "check_vitals",
    "map":                      "check_vitals",
    "hypotension":              "check_vitals",
    "hypertension":             "check_vitals",
    # Heart rate and rhythm
    "heart rate":               "check_vitals",
    "hr":                       "check_vitals",
    "pulse":                    "check_vitals",
    "tachycardia":              "check_vitals",
    "bradycardia":              "check_vitals",
    "rhythm":                   "check_vitals",
    "ecg rhythm":               "check_vitals",
    # Respiratory
    "respiratory rate":         "check_vitals",
    "rr":                       "check_vitals",
    "breathing rate":           "check_vitals",
    "respiratory":              "check_vitals",
    # Oxygen saturation
    "oxygen saturation":        "check_vitals",
    "spo2":                     "check_vitals",
    "o2 sat":                   "check_vitals",
    "oxygen":                   "check_vitals",
    "saturation":               "check_vitals",
    "oxygenation":              "check_vitals",
    "hypoxia":                  "check_vitals",
    # Temperature
    "temperature":              "check_vitals",
    "temp":                     "check_vitals",
    "fever":                    "check_vitals",
    "pyrexia":                  "check_vitals",
    "hypothermia":              "check_vitals",
    # Neurological/conscious level
    "gcs":                      "check_vitals",
    "glasgow":                  "check_vitals",
    "conscious":                "check_vitals",
    "consciousness":            "check_vitals",
    "level of consciousness":   "check_vitals",
    "loc":                      "check_vitals",
    "avpu":                     "check_vitals",
    "pupils":                   "check_vitals",
    "neurological":             "check_vitals",
    # Point-of-care values commonly obtained at vital assessment
    "glucose":                  "check_vitals",
    "blood sugar":              "check_vitals",
    "bsl":                      "check_vitals",
    "capillary glucose":        "check_vitals",
    # Repeat and trend language
    "repeat vitals":            "check_vitals",
    "repeat observations":      "check_vitals",
    "trend":                    "check_vitals",
    "trending":                 "check_vitals",
    "follow up vitals":         "check_vitals",
    "current obs":              "check_vitals",
    "current vitals":           "check_vitals",
    "latest vitals":            "check_vitals",
    # Capillary refill and perfusion (often assessed with vitals)
    "capillary refill":         "check_vitals",
    "crt":                      "check_vitals",
    "perfusion":                "check_vitals",

    # ── examine_patient ───────────────────────────────────────────────────────
    # General examination language
    "exam":                     "examine_patient",
    "examination":              "examine_patient",
    "examine":                  "examine_patient",
    "physical":                 "examine_patient",
    "physical exam":            "examine_patient",
    "physical examination":     "examine_patient",
    "clinical exam":            "examine_patient",
    "clinical examination":     "examine_patient",
    "clinical findings":        "examine_patient",
    "clinical signs":           "examine_patient",
    "findings":                 "examine_patient",
    # Inspection and appearance
    "inspect":                  "examine_patient",
    "inspection":               "examine_patient",
    "appearance":               "examine_patient",
    "look":                     "examine_patient",
    "skin":                     "examine_patient",
    "colour":                   "examine_patient",
    "color":                    "examine_patient",
    "rash":                     "examine_patient",
    "petechiae":                "examine_patient",
    "bruising":                 "examine_patient",
    "cyanosis":                 "examine_patient",
    "jaundice":                 "examine_patient",
    "pallor":                   "examine_patient",
    "diaphoresis":              "examine_patient",
    # Palpation
    "palpate":                  "examine_patient",
    "palpation":                "examine_patient",
    "palpating":                "examine_patient",
    "tenderness":               "examine_patient",
    "guarding":                 "examine_patient",
    "rigidity":                 "examine_patient",
    "mass":                     "examine_patient",
    "lump":                     "examine_patient",
    "swelling":                 "examine_patient",
    "oedema":                   "examine_patient",
    "edema":                    "examine_patient",
    "crepitus":                 "examine_patient",
    # Auscultation
    "auscultate":               "examine_patient",
    "auscultation":             "examine_patient",
    "listen":                   "examine_patient",
    "breath sounds":            "examine_patient",
    "heart sounds":             "examine_patient",
    "bowel sounds":             "examine_patient",
    "murmur":                   "examine_patient",
    "wheeze":                   "examine_patient",
    "crackles":                 "examine_patient",
    "crepitations":             "examine_patient",
    "rhonchi":                  "examine_patient",
    # Percussion
    "percussion":               "examine_patient",
    "percuss":                  "examine_patient",
    "dullness":                 "examine_patient",
    # System-specific focused examination
    "abdominal":                "examine_patient",
    "abdomen":                  "examine_patient",
    "chest":                    "examine_patient",
    "respiratory exam":         "examine_patient",
    "cardiac exam":             "examine_patient",
    "cardiac":                  "examine_patient",
    "cardiovascular":           "examine_patient",
    "neuro exam":               "examine_patient",
    "neurological exam":        "examine_patient",
    "focused exam":             "examine_patient",
    "targeted exam":            "examine_patient",
    # Specific neurological signs
    "kernig":                   "examine_patient",
    "brudzinski":               "examine_patient",
    "meningism":                "examine_patient",
    "focal":                    "examine_patient",
    "focal neurology":          "examine_patient",
    "power":                    "examine_patient",
    "reflexes":                 "examine_patient",
    "plantar":                  "examine_patient",
    "drift":                    "examine_patient",
    "pronator drift":           "examine_patient",
    # Wound and skin findings
    "wound":                    "examine_patient",
    "wound exam":               "examine_patient",
    "leg":                      "examine_patient",
    "ulcer":                    "examine_patient",
    "discharge":                "examine_patient",
    "necrosis":                 "examine_patient",
    # Specific embolic/vascular signs
    "osler":                    "examine_patient",
    "janeway":                  "examine_patient",
    "splinter":                 "examine_patient",
    "fundoscopy":               "examine_patient",
    "roth":                     "examine_patient",
    # General red-flag language
    "red flag":                 "examine_patient",
    "red flags":                "examine_patient",
    "signs":                    "examine_patient",
    "sign":                     "examine_patient",
}
"""
Keyword → trigger lookup table.

Algorithm (shared with InfoRevealer and score_clinical_path):
    For each keyword present in the lowercase question text:
        score[trigger] += len(keyword)
    Winner = argmax(score), ties broken alphabetically.

Longer, more specific phrases score higher than short generic words,
which is the intended behaviour — "bilateral bp" beats "bp" on a
bilateral BP question.
"""