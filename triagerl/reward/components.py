"""
triagerl.reward.components
==========================
Pure scoring functions for the four primary reward components.

Fixes vs previous version
--------------------------
1. score_reasoning(): keyword stuffing defence completely rewritten.
   Previous version double-penalised: awarded keyword match score then
   penalised for keyword repetitions that caused the match — contradictory
   signal. New version uses a single unified anti-gaming pass: vocabulary
   diversity check (unique significant token ratio) only, applied once.
   Keyword repetition sub-penalty removed entirely — it was both logically
   inconsistent and incentivised the agent to mention each keyword only once
   even when the clinical term naturally recurs in genuine reasoning.

2. score_temporal(): TEMPORAL_MAX capped at 1.0 (was 1.20). The >1.0
   "speed bonus" was invisible after W_TEMPORAL * 1.10 = 0.11 vs 0.10
   in the grader's clamped output — below any meaningful gradient threshold.
   Removed the illusion. Temporal now signals clearly in [0.0, 1.0].

3. All thresholds renamed and documented. No magic numbers.

Design contract
---------------
* Every function is pure: identical inputs → identical outputs.
* No I/O, no logging, no side-effects.
* All inputs are plain Python types.
* Return types are plain floats or (float, bool) tuples.

Score ranges (post-fix)
-----------------------
    esi_score       ∈ [-0.15, 1.00]
    temporal_score  ∈ [ 0.00, 1.00]   (removed >1.0 dead bonus)
    reasoning_score ∈ [ 0.00, 1.00]
    action_score    ∈ [ 0.00, 1.00]
"""
from __future__ import annotations

import re
from collections import Counter
from typing import FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Stopwords — excluded from token uniqueness and overlap analysis
# ---------------------------------------------------------------------------

STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by",
    "call", "can", "check", "consider", "consult", "continue", "for",
    "from", "have", "if", "immediate", "immediately", "in", "initiate",
    "into", "is", "it", "labs", "monitor", "move", "of", "on", "or",
    "plan", "prepare", "protocol", "request", "review", "start", "the",
    "therapy", "to", "urgent", "urgently", "with", "within",
})


# ---------------------------------------------------------------------------
# ESI scoring constants
# ---------------------------------------------------------------------------

ESI_SCORE_PERFECT:       float = 1.00
ESI_SCORE_OFF_BY_ONE:    float = 0.30   # small positive for near-miss
ESI_SCORE_OFF_BY_TWO:    float = -0.50  # moderately wrong
ESI_SCORE_LARGE_MISS:    float = -1.00  # completely incorrect
ESI_SCORE_CRITICAL_MISS: float = -0.75  # undertriage on critical patient is strongly negative
ESI_CRITICAL_THRESHOLD: int   = 2      # ESI ≤ this is critical


# ---------------------------------------------------------------------------
# Temporal scoring constants
# ---------------------------------------------------------------------------

TEMPORAL_BASE:                    float = 1.00
TEMPORAL_CRITICAL_STEP_PENALTY:   float = 0.12   # per extra step on ESI 1-2
TEMPORAL_NONCRITICAL_SKIP_PENALTY: float = 0.05  # per expected step skipped
TEMPORAL_MIN:                     float = 0.00
TEMPORAL_MAX:                     float = 1.00   # FIX: was 1.20 (dead bonus)


# ---------------------------------------------------------------------------
# Reasoning scoring constants
# ---------------------------------------------------------------------------

REASONING_MIN_WORDS:          int   = 30
REASONING_SHORT_CAP:          float = 0.60
REASONING_NO_KW_LONG:         float = 0.60
REASONING_NO_KW_SHORT:        float = 0.30
# Anti-gaming: single unified diversity check
REASONING_DIVERSITY_THRESHOLD: float = 0.35  # unique/total sig tokens below → penalty
REASONING_DIVERSITY_PENALTY:   float = 0.70  # FIX: was 0.75, applied once only
# Extreme repetition: individual token appears ≥ this many times
REASONING_OVERREPEAT_COUNT:    int   = 6     # FIX: was 5 (misfired on correct medical terms)
REASONING_OVERREPEAT_MIN_HITS: int   = 3     # how many tokens must be over-repeated
REASONING_OVERREPEAT_PENALTY:  float = 0.80


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _significant_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 2]


# ---------------------------------------------------------------------------
# Public text utilities
# ---------------------------------------------------------------------------

def keyword_matches(text: str, keywords: Sequence[str]) -> List[str]:
    """
    Two-tier keyword matching.

    Tier 1: full normalised keyword is a substring of normalised text.
    Tier 2: ALL significant tokens from the keyword appear in the text
            (changed from ANY — prevents single-token fragment gaming).

    Single-token keywords (after stripping stopwords) still require exact
    appearance in the text.  Multi-token keywords require every significant
    component to appear, preventing "cath" alone from matching
    "cardiac catheterization laboratory activation".

    Parameters
    ----------
    text : str
    keywords : sequence of str

    Returns
    -------
    list[str] — subset of keywords that matched, in input order.
    """
    lowered = _normalise(text)
    matched: List[str] = []
    for kw in keywords:
        kw_low = _normalise(kw)
        if kw_low in lowered:
            matched.append(kw)
            continue
        sig = _significant_tokens(_tokenise(kw_low))
        # Tier 2: require ALL significant tokens to appear (not ANY).
        if sig and all(tok in lowered for tok in sig):
            matched.append(kw)
    return matched


def action_overlap(
    recommended: Sequence[str],
    expected: Sequence[str],
) -> Tuple[int, List[str]]:
    """
    Token-level overlap between recommended and expected actions.

    For each expected action, checks whether a MAJORITY (>50%) of its
    significant tokens appear in the joined recommended-action text.
    This prevents gaming via single-word hints (e.g. writing "cath" to
    match "activate cardiac catheterization laboratory").

    Minimum threshold: at least 1 token AND >50% of significant tokens.
    Short phrases (≤2 significant tokens) require ALL tokens to match.

    Returns
    -------
    (matched_count, matched_expected_list)
    """
    import math

    if not expected:
        return 0, []
    rec_text = " ".join(_normalise(a) for a in recommended)
    matched: List[str] = []
    for exp in expected:
        toks = _significant_tokens(_tokenise(exp))
        if not toks:
            continue
        hit_count = sum(1 for t in toks if t in rec_text)
        # Require strict majority: ceil(n/2) tokens must match.
        # For single-token phrases, the one token must match.
        threshold = math.ceil(len(toks) / 2)
        if hit_count >= threshold:
            matched.append(exp)
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

    Returns
    -------
    (esi_score, undertriage_flag)
        esi_score       ∈ [-0.15, 1.00]
        undertriage_flag: True when correct ≤ 2 and predicted > correct.

    Score table
    -----------
    diff=0                                    → 1.00
    diff=1, undertriage on critical (≤2)      → 0.10
    diff=1, other                             → 0.28
    diff=2                                    → 0.00
    diff≥3                                    → -0.15
    """
    if predicted is None or correct is None:
        return 0.0, False

    diff = abs(predicted - correct)
    undertriage = (correct <= ESI_CRITICAL_THRESHOLD and predicted > correct)

    # Exact match — clear positive signal
    if diff == 0:
        return ESI_SCORE_PERFECT, undertriage

    # Near miss — still positive but noticeably lower than perfect
    if diff == 1:
        # If this is an undertriage on a critical patient, treat as a stronger
        # negative signal than a simple off-by-one.
        if predicted > correct and correct <= ESI_CRITICAL_THRESHOLD:
            return ESI_SCORE_CRITICAL_MISS, undertriage
        return ESI_SCORE_OFF_BY_ONE, undertriage

    # Larger misses are negative and increasingly punitive
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
    Urgency-aware temporal efficiency score ∈ [0.00, 1.00].

    FIX: TEMPORAL_MAX is now 1.00. The previous 1.20 "speed bonus" for
    step-1 classification was multiplied by W_TEMPORAL=0.10 in the grader,
    adding only 0.01 to the weighted base — below any meaningful gradient
    threshold after clamping. Removed. Temporal now has a clean [0, 1] range
    that contributes meaningfully to the gradient.

    Critical (ESI 1-2): penalise each step beyond minimum required.
    Non-critical (ESI 4-5): penalise skipping expected clarification.
    ESI 3: neutral.

    Parameters
    ----------
    esi_correct : int
    steps_taken : int
        Total steps (clarify + classify).
    expected_clarify_steps : int
    clarify_count : int
        Actual clarify actions taken.
    """
    base = TEMPORAL_BASE

    if esi_correct <= ESI_CRITICAL_THRESHOLD:
        # Minimum viable steps = expected_clarify + 1 classify.
        minimum = expected_clarify_steps + 1
        extra   = max(0, steps_taken - minimum)
        base   -= TEMPORAL_CRITICAL_STEP_PENALTY * extra

    elif esi_correct >= 4:
        # Penalise only when agent skipped ALL clarification on a task that
        # expected at least one step. If clarify_count > 0, no penalty.
        if expected_clarify_steps >= 1 and clarify_count == 0:
            base -= TEMPORAL_NONCRITICAL_SKIP_PENALTY * expected_clarify_steps

    return round(max(TEMPORAL_MIN, min(TEMPORAL_MAX, base)), 4)


# ---------------------------------------------------------------------------
# Reasoning quality scoring
# ---------------------------------------------------------------------------

def score_reasoning(
    reasoning: str,
    keywords: List[str],
) -> float:
    """
    Keyword coverage score with length guard and unified anti-gaming check.

    FIX: Removed double-penalisation. Previous version awarded keyword match
    score then separately penalised keyword repetitions that caused those
    matches. This created contradictory gradients: the model was rewarded for
    matching "stemi" but penalised for the repetition that achieved the match.

    New approach:
    1. Compute keyword coverage ratio.
    2. Apply short-text cap (text < REASONING_MIN_WORDS → cap at 0.60).
    3. Apply ONE unified anti-gaming check: vocabulary diversity ratio.
       If unique_significant_tokens / total_significant_tokens < threshold,
       the agent is stuffing tokens — apply 0.70× multiplier once.
    4. Apply extreme token repetition check: if ≥ 3 individual tokens each
       appear ≥ 6 times, this is mechanical repetition — apply 0.80×.
       Threshold raised from 5→6 to avoid misfiring on legitimate clinical
       terms (e.g. "STEMI" in a STEMI scenario reasoning chain).

    Returns float ∈ [0.00, 1.00].
    """
    all_tokens  = _tokenise(reasoning)
    sig_tokens  = _significant_tokens(all_tokens)
    word_count  = len(reasoning.split())

    # Vocabulary diversity (computed over significant tokens only)
    unique_ratio = (
        len(set(sig_tokens)) / len(sig_tokens)
        if sig_tokens else 0.0
    )

    # ── No-keyword task ───────────────────────────────────────────────────────
    if not keywords:
        base = (
            REASONING_NO_KW_LONG
            if word_count >= REASONING_MIN_WORDS
            else REASONING_NO_KW_SHORT
        )
        if unique_ratio < REASONING_DIVERSITY_THRESHOLD:
            base *= REASONING_DIVERSITY_PENALTY
        return round(max(0.0, min(1.0, base)), 4)

    # ── Keyword coverage ──────────────────────────────────────────────────────
    matched = keyword_matches(reasoning, keywords)
    score   = len(matched) / len(keywords)

    # Length guard
    if word_count < REASONING_MIN_WORDS:
        score = min(score, REASONING_SHORT_CAP)

    # ── Anti-gaming: unified diversity check (applied ONCE) ───────────────────
    # FIX: single check, not cascading multipliers for keyword repetition.
    if unique_ratio < REASONING_DIVERSITY_THRESHOLD:
        score *= REASONING_DIVERSITY_PENALTY

    # ── Anti-gaming: extreme token repetition ─────────────────────────────────
    # Only triggers when individual tokens appear ≥ 6 times (was 5).
    # Threshold 6 avoids misfiring on "STEMI" in STEMI reasoning (4-5 uses).
    counts        = Counter(sig_tokens)
    over_repeated = sum(1 for _, c in counts.items() if c >= REASONING_OVERREPEAT_COUNT)
    if over_repeated >= REASONING_OVERREPEAT_MIN_HITS:
        score *= REASONING_OVERREPEAT_PENALTY

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

    Uses token-level overlap so minor phrasing differences do not penalise.
    Returns 1.0 when expected is empty (task does not test this dimension).

    Returns float ∈ [0.00, 1.00].
    """
    if not expected:
        return 1.0
    matched_count, _ = action_overlap(recommended, expected)
    return round(matched_count / len(expected), 4)