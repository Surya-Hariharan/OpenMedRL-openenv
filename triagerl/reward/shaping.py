"""Reward shaping constants and helpers.

The legacy environment currently applies shaping inline in the environment
step loop; this module provides a stable home for the shaping policy values.
"""

CLARIFY_NEW_INFO_BONUS = 0.02
CLARIFY_REPEATED_PENALTY = -0.01
CLARIFY_NO_NEED_PENALTY = -0.02
RUSH_CLASSIFY_PENALTY = -0.05
TIMEOUT_PENALTY = -0.10
