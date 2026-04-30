"""
triagerl.reward.components
==========================
Pure scoring functions for the four primary reward components.

Design contract
---------------
*  Every function is pure: identical inputs → identical outputs.
*  No I/O, no logging, no side-effects.
*  No magic numbers — all thresholds are named module-level constants.
*  All inputs are plain Python types (str, int, float, list).
   No Pydantic models, no TaskConfig — the grader layer handles conversion.
*  Return types are plain (float,) or (float, bool) tuples.
   The grader assembles them into RewardBreakdown.

Score ranges
------------
    esi_score       ∈ [-0.15, 1.00]
    temporal_score  ∈ [ 0.00, 1.20]  (>1.0 = critical-patient speed bonus)
    reasoning_score ∈ [ 0.00, 1.00]
    action_score    ∈ [ 0.00, 1.00]

Anti-gaming measures
--------------------
*  Reasoning: token-uniqueness ratio < 0.35 triggers a 0.75× penalty
   (keyword stuffing).  Tokens appearing ≥5 times trigger a 0.80× penalty.
   Keyword repetitions > 2 occurrences each trigger an additional 0.75×.
*  ESI: undertriage on a critical patient (correct ≤ 2, predicted > correct)
   earns only 0.10 at diff=1 vs 0.28 for over-triage — asymmetric safety bias.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Named constants — no magic numbers below this line
# ---------------------------------------------------------------------------

# ── ESI scoring table ────────────────────────────────────────────────────────
ESI_SCORE_PERFECT:          float = 1.00
ESI_SCORE_OFF_BY_ONE:       float = 0.28   # overtriage or non-critical undertriage
ESI_SCORE_CRITICAL_MISS:    float = 0.10   # diff=1 undertriage on ESI ≤ 2
ESI_SCORE_OFF_BY_TWO:       float = 0.00
ESI_SCORE_LARGE_MISS:       float = -0.15  # diff ≥ 3
ESI_CRITICAL_THRESHOLD:     int   = 2      # ESI ≤ this → critical case

# ── Temporal scoring ─────────────────────────────────────────────────────────
TEMPORAL_BASE:              float = 1.00
TEMPORAL_CRITICAL_STEP_PENALTY: float = 0.10  # per extra step on ESI 1-2
TEMPORAL_CRITICAL_SPEED_BONUS:  float = 0.10  # classify on step 1 for ESI 1-2
TEMPORAL_NONCRITICAL_SKIP_PENALTY: float = 0.04  # per expected step skipped
TEMPORAL_MIN:               float = 0.00
TEMPORAL_MAX:               float = 1.20   # >1.0 = bonus territory

# ── Reasoning scoring ────────────────────────────────────────────────────────
REASONING_MIN_WORDS:        int   = 30     # minimum words for full keyword credit
REASONING_NO_KEYWORDS_LONG: float = 0.60  # base score when task has no keywords, text ≥ 30 words
REASONING_NO_KEYWORDS_SHORT: float = 0.30 # base score when task has no keywords, text < 30 words
REASONING_SHORT_CAP:        float = 0.60  # max score when text < MIN_WORDS
REASONING_UNIQUE_RATIO_THRESHOLD: float = 0.35   # below → keyword stuffing
REASONING_UNIQUE_RATIO_PENALTY:   float = 0.75   # multiplier for stuffing
REASONING_OVERREPEAT_COUNT:       int   = 5      # token count that triggers over-repeat flag
REASONING_OVERREPEAT_THRESHOLD:   int   = 3      # number of over-repeated tokens to trigger penalty
REASONING_OVERREPEAT_PENALTY:     float = 0.80   # multiplier
REASONING_KEYWORD_REPEAT_EXCESS:  int   = 2      # allowed occurrences before penalty
REASONING_KEYWORD_REPEAT_TRIGGER: int   = 4      # total excess reps to trigger penalty
REASONING_KEYWORD_REPEAT_PENALTY: float = 0.75   # multiplier

# ── Stopwords (excluded from token uniqueness and overlap analysis) ───────────
STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by",
    "call", "can", "check", "consider", "consult", "continue", "for",
    "from", "have", "if", "immediate", "immediately", "in", "initiate",
    "into", "is", "it", "labs", "monitor", "move", "of", "on", "or",
    "plan", "prepare", "protocol", "request", "review", "start", "the",
    "therapy", "to", "urgent", "urgently", "with", "within",
})


# ---------------------------------------------------------------------------
# Text utility functions (private — used only within this module)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenise(text: str) -> List[str]:
    """Extract alphanumeric tokens in lowercase."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _significant_tokens(tokens: List[str]) -> List[str]:
    """Filter out stopwords, digits, and very short tokens."""
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 2]


# ---------------------------------------------------------------------------
# Public text utilities (re-exported for use by grader and path_quality)
# ---------------------------------------------------------------------------

def keyword_matches(text: str, keywords: Sequence[str]) -> List[str]:
    """
    Flexible keyword matching with two-tier OR logic.

    A keyword matches if:
      1. The full normalised keyword string is a substring of the
         normalised text (exact substring match), OR
      2. Any significant (non-stopword, len > 2, non-digit) token from the
         keyword appears in the normalised text.

    Tier-2 exists to handle paraphrasing: "history of past illness" matches
    the keyword "past history" via the token "past".

    Parameters
    ----------
    text : str
        The text to search (agent reasoning, recommended action string, etc.)
    keywords : sequence of str
        Keywords to look for.

    Returns
    -------
    list[str]
        Subset of ``keywords`` that matched, in input order.
    """
    lowered = _normalise(text)
    matched: List[str] = []

    for kw in keywords:
        kw_low = _normalise(kw)
        # Tier 1: exact substring
        if kw_low in lowered:
            matched.append(kw)
            continue
        # Tier 2: any significant token appears
        sig = _significant_tokens(_tokenise(kw_low))
        if sig and any(tok in lowered for tok in sig):
            matched.append(kw)

    return matched


def action_overlap(
    recommended: Sequence[str],
    expected: Sequence[str],
) -> Tuple[int, List[str]]:
    """
    Token-level overlap between recommended and expected actions.

    For each expected action, checks whether any significant token from
    that action appears in the joined recommended-action text.

    Returns
    -------
    (matched_count, matched_expected_list)
    """
    if not expected:
        return 0, []

    rec_text = " ".join(_normalise(a) for a in recommended)
    matched: List[str] = []

    for exp_action in expected:
        toks = _significant_tokens(_tokenise(exp_action))
        if not toks:
            continue
        if any(t in rec_text for t in toks):
            matched.append(exp_action)

    return len(matched), matched


# ---------------------------------------------------------------------------
# ESI accuracy scoring
# ---------------------------------------------------------------------------

def score_esi(
    predicted: Optional[int],
    correct: Optional[int],
) -> Tuple[float, bool]:
    """
    Compute ESI accuracy score and undertriage flag.

    Parameters
    ----------
    predicted : int | None
        The ESI level the agent assigned.
    correct : int | None
        The ground-truth ESI level for this task.

    Returns
    -------
    (esi_score, undertriage_flag)
        esi_score       ∈ [-0.15, 1.00]
        undertriage_flag is True when ``correct ≤ 2`` and ``predicted > correct``.

    Score table
    -----------
    diff=0              → 1.00  (perfect)
    diff=1, undertriage on critical (correct ≤ 2) → 0.10
    diff=1, any other   → 0.28
    diff=2              → 0.00
    diff≥3              → -0.15
    """
    if predicted is None or correct is None:
        return 0.0, False

    diff        = abs(predicted - correct)
    undertriage = (correct <= ESI_CRITICAL_THRESHOLD and predicted > correct)

    if diff == 0:
        return ESI_SCORE_PERFECT, undertriage

    if diff == 1:
        # Asymmetric: undertriaging a critical patient is penalised more
        # than over-triaging (which merely wastes resources).
        if predicted > correct and correct <= ESI_CRITICAL_THRESHOLD:
            return ESI_SCORE_CRITICAL_MISS, undertriage
        return ESI_SCORE_OFF_BY_ONE, undertriage

    if diff == 2:
        return ESI_SCORE_OFF_BY_TWO, undertriage

    # diff >= 3
    return ESI_SCORE_LARGE_MISS, undertriage


# ---------------------------------------------------------------------------
# Temporal efficiency scoring
# ---------------------------------------------------------------------------

def score_temporal(
    esi_correct: int,
    steps_taken: int,
    expected_clarify_steps: int,
    clarify_count: int = 0,
) -> float:
    """
    Urgency-aware temporal efficiency score.

    Design rationale
    ----------------
    Critical patients (ESI 1-2) deteriorate with time, so extra steps are
    penalised heavily and immediate correct classification is bonused.

    Low-acuity patients (ESI 4-5) require at least some assessment before
    classification — skipping clarification entirely when the task expected
    it earns a small penalty.  But once any clarification has been done,
    being quick to classify is not penalised.

    ESI 3 is neutral — no bonus, no penalty.

    Parameters
    ----------
    esi_correct : int
        Ground-truth ESI level (1–5).
    steps_taken : int
        Total steps taken in the episode (clarify + classify).
    expected_clarify_steps : int
        Number of clarify steps the task designer expected.
    clarify_count : int
        Actual number of clarify actions taken (default 0 for single-action
        callers like ``grade()``).

    Returns
    -------
    float
        Score ∈ [0.00, 1.20].  Values > 1.0 are a speed bonus.
    """
    base = TEMPORAL_BASE

    if esi_correct <= ESI_CRITICAL_THRESHOLD:
        # Heavy penalty for dawdling on a critical patient.
        extra_steps = max(0, steps_taken - (expected_clarify_steps + 1))
        base -= TEMPORAL_CRITICAL_STEP_PENALTY * extra_steps
        # Bonus for classify-on-step-1 (correct triage at first glance).
        if steps_taken == 1:
            base += TEMPORAL_CRITICAL_SPEED_BONUS

    elif esi_correct >= 4:
        # Low urgency: penalise only when the agent skipped all clarification
        # on a task that expected at least one step.  If clarify_count > 0,
        # the agent already assessed — no additional penalty.
        if expected_clarify_steps >= 1 and clarify_count == 0:
            base -= TEMPORAL_NONCRITICAL_SKIP_PENALTY * expected_clarify_steps

    # ESI 3: neutral — no modification to base.

    return round(max(TEMPORAL_MIN, min(TEMPORAL_MAX, base)), 4)


# ---------------------------------------------------------------------------
# Reasoning quality scoring
# ---------------------------------------------------------------------------

def score_reasoning(
    reasoning: str,
    keywords: List[str],
) -> float:
    """
    Keyword coverage score with length guard and anti-gaming penalties.

    Algorithm
    ---------
    1. Compute token uniqueness ratio to detect keyword stuffing.
    2. If the task has no keywords: return a fixed base score scaled
       by length, with uniqueness penalty applied.
    3. Otherwise: compute keyword match ratio via ``keyword_matches()``.
    4. Apply length cap (short text ≤ ``REASONING_SHORT_CAP``).
    5. Apply anti-gaming multipliers in sequence:
        * Low uniqueness ratio         → 0.75×
        * Too many over-repeated tokens → 0.80×
        * Keyword repetition stuffing  → 0.75×

    Parameters
    ----------
    reasoning : str
        The agent's reasoning chain text.
    keywords : list[str]
        Task-specific clinical keywords from ``TaskConfig.key_reasoning_keywords``.

    Returns
    -------
    float
        Score ∈ [0.00, 1.00], rounded to 4 decimal places.
    """
    all_tokens  = _tokenise(reasoning)
    sig_tokens  = [t for t in all_tokens if t not in STOPWORDS]
    unique_ratio = (
        len(set(sig_tokens)) / len(sig_tokens)
        if sig_tokens else 0.0
    )
    word_count = len(reasoning.split())

    # ── No-keyword task ───────────────────────────────────────────────────────
    if not keywords:
        base = (
            REASONING_NO_KEYWORDS_LONG
            if word_count >= REASONING_MIN_WORDS
            else REASONING_NO_KEYWORDS_SHORT
        )
        if unique_ratio < REASONING_UNIQUE_RATIO_THRESHOLD:
            base *= REASONING_UNIQUE_RATIO_PENALTY
        return round(base, 4)

    # ── Keyword coverage ──────────────────────────────────────────────────────
    matched = keyword_matches(reasoning, keywords)
    score   = len(matched) / len(keywords)   # ∈ [0.0, 1.0]

    # Length guard: short reasoning can only earn up to the cap.
    if word_count < REASONING_MIN_WORDS:
        score = min(score, REASONING_SHORT_CAP)

    # ── Anti-gaming multipliers ───────────────────────────────────────────────
    counts       = Counter(sig_tokens)
    over_repeated = sum(1 for _, c in counts.items() if c >= REASONING_OVERREPEAT_COUNT)

    # Keyword repetition excess: count uses of each matched keyword beyond 2.
    keyword_lower_hits = [kw.lower() for kw in matched]
    normalised_text    = _normalise(reasoning)
    keyword_repeat_excess = sum(
        max(0, normalised_text.count(kw) - REASONING_KEYWORD_REPEAT_EXCESS)
        for kw in keyword_lower_hits
    )

    if unique_ratio < REASONING_UNIQUE_RATIO_THRESHOLD:
        score *= REASONING_UNIQUE_RATIO_PENALTY

    if over_repeated >= REASONING_OVERREPEAT_THRESHOLD:
        score *= REASONING_OVERREPEAT_PENALTY

    if keyword_repeat_excess >= REASONING_KEYWORD_REPEAT_TRIGGER:
        score *= REASONING_KEYWORD_REPEAT_PENALTY

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Action coverage scoring
# ---------------------------------------------------------------------------

def score_actions(
    recommended: List[str],
    expected: List[str],
) -> float:
    """
    Fraction of expected actions covered by recommended actions.

    Uses token-level overlap (``action_overlap``) so minor phrasing
    differences do not penalise the agent.

    When ``expected`` is empty the task does not test this dimension —
    return 1.0 (neutral, does not inflate or deflate final score).

    Parameters
    ----------
    recommended : list[str]
        Actions the agent recommended.
    expected : list[str]
        Gold-standard actions from ``TaskConfig.expected_actions``.

    Returns
    -------
    float
        Score ∈ [0.00, 1.00], rounded to 4 decimal places.
    """
    if not expected:
        return 1.0

    matched_count, _ = action_overlap(recommended, expected)
    return round(matched_count / len(expected), 4)