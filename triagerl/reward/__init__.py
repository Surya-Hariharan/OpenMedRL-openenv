"""
triagerl.reward
===============
Reward computation layer — pure functions, no I/O, no env coupling.

Public surface
--------------

Primary scoring (episode-end, RL training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compute_final_score(action, task, action_history, steps_taken)
        → (float, ComponentDict, str)

    build_episode_metrics(session_id, task, action_history, final_reward,
                          breakdown, extra_metrics)
        → EpisodeMetrics

Clarify shaping (per-step, RL training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compute_clarify_shaping(question, reveal_payload, clarify_count,
                             task, clarify_penalty_dampening)
        → ClarifyShapingResult

    should_advance_to_disposition(clarify_count, expected_clarify_steps)
        → bool

Component scorers (testing / ablation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    score_esi(predicted, correct)         → (float, bool)
    score_temporal(esi_correct, steps, expected_steps, clarify_count)  → float
    score_reasoning(reasoning, keywords)   → float
    score_actions(recommended, expected)   → float
    score_clinical_path(action_history, task)  → float

Safety modifier
~~~~~~~~~~~~~~~
    apply_safety_modifier(raw, undertriage, multiplier)
        → (adjusted_float, effective_factor_float)
    is_undertriage(correct_esi, predicted_esi) → bool

Text utilities
~~~~~~~~~~~~~~
    keyword_matches(text, keywords)  → list[str]
    action_overlap(recommended, expected)  → (int, list[str])

LLM judge (offline evaluation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grade_with_llm(action, task, episode_summary, model)     [async]
    grade_with_llm_safe(action, task, ..., fallback)         [async]
    make_keyword_fallback(grader_fn)                         → FallbackFn

    LLMJudgeResult
    LLMJudgeError / LLMUnavailableError / LLMAPIError / LLMParseError

Legacy (backward-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grade(action, task) → TriageReward

Dependency graph (no cycles)
-----------------------------
    triagerl.core.constants   ← path_quality (KEYWORD_TO_TRIGGER)
    triagerl.core.types       ← grader (RewardComponent)
    triagerl.core.models      ← grader, llm_judge
    triagerl.tasks            ← path_quality, grader, shaping

    reward/components.py   (pure math)
    reward/safety.py       (pure math)
    reward/path_quality.py (imports KEYWORD_TO_TRIGGER from core.constants)
    reward/shaping.py      (clarify shaping, returns ClarifyShapingResult)
    reward/grader.py       (orchestrates all scorers)
    reward/llm_judge.py    (async, offline only)
"""
from .components import (
    STOPWORDS,
    action_overlap,
    keyword_matches,
    score_actions,
    score_esi,
    score_reasoning,
    score_temporal,
)
from .grader import (
    ComponentDict,
    build_episode_metrics,
    compute_final_score,
    grade,
)
from .llm_judge import (
    LLMAPIError,
    LLMJudgeError,
    LLMJudgeResult,
    LLMParseError,
    LLMUnavailableError,
    grade_with_llm,
    grade_with_llm_safe,
    make_keyword_fallback,
)
from .path_quality import count_useful_clarifications, score_clinical_path
from .safety import (
    UNDERTRIAGE_MULTIPLIER,
    apply_safety_modifier,
    apply_safety_modifier_to_components,
    is_undertriage,
)
from .shaping import (
    TIMEOUT_FEEDBACK,
    TIMEOUT_PENALTY,
    ClarifyShapingResult,
    compute_clarify_shaping,
    should_advance_to_disposition,
)

__all__ = [
    # Primary scorers
    "compute_final_score",
    "build_episode_metrics",
    # Clarify shaping
    "compute_clarify_shaping",
    "ClarifyShapingResult",
    "should_advance_to_disposition",
    "TIMEOUT_PENALTY",
    "TIMEOUT_FEEDBACK",
    # Component scorers
    "score_esi",
    "score_temporal",
    "score_reasoning",
    "score_actions",
    "score_clinical_path",
    "count_useful_clarifications",
    # Safety
    "apply_safety_modifier",
    "apply_safety_modifier_to_components",
    "is_undertriage",
    "UNDERTRIAGE_MULTIPLIER",
    # Text utilities
    "keyword_matches",
    "action_overlap",
    "STOPWORDS",
    # LLM judge
    "grade_with_llm",
    "grade_with_llm_safe",
    "make_keyword_fallback",
    "LLMJudgeResult",
    "LLMJudgeError",
    "LLMUnavailableError",
    "LLMAPIError",
    "LLMParseError",
    # Legacy
    "grade",
    "ComponentDict",
]