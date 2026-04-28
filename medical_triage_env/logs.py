"""
Structured logging configuration for medical triage environment.

Provides consistent JSON logging in production and human-readable console logging
in development mode. Never logs PHI (Protected Health Information).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import structlog
from structlog.types import Processor


def _scrub_phi(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or mask potential PHI from logs."""
    if "patient_name" in event_dict:
        event_dict["patient_name"] = "[REDACTED]"
    
    if "patient_id" in event_dict:
        pid = str(event_dict["patient_id"])
        if len(pid) > 8:
            event_dict["patient_id"] = f"{pid[:8]}..."
    
    return event_dict


def _configure_structlog() -> None:
    """Configure structlog based on environment."""
    env = os.getenv("ENV", "production").lower()
    
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _scrub_phi,
    ]
    
    if env == "development":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(10),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


_configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        BoundLogger instance for structured logging
    """
    return structlog.get_logger(name)


def log_episode_metrics(metrics: Any) -> None:
    """Emit standardized episode telemetry for RL training dashboards."""
    logger = get_logger("medical_triage_env.telemetry")
    payload = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)

    reward_breakdown = payload.get("reward_breakdown", {}) or {}
    # Default to redacting internal task ids in logs (HF Spaces stdout is often public).
    # Set ENV=development to keep full ids during local debugging.
    env = os.getenv("ENV", "production").lower()
    task_id_logged = payload.get("task_id") if env == "development" else "[REDACTED]"
    logger.info(
        "episode_metrics",
        ts=datetime.now(timezone.utc).isoformat(),
        session_id=payload.get("session_id"),
        task_id=task_id_logged,
        difficulty=payload.get("difficulty"),
        category=payload.get("category"),
        esi_correct=payload.get("esi_correct"),
        esi_predicted=payload.get("esi_predicted"),
        steps_taken=payload.get("steps_taken"),
        max_steps=payload.get("max_steps"),
        total_reward=payload.get("total_reward"),
        undertriage=payload.get("undertriage"),
        overtriage=payload.get("overtriage"),
        clarification_count=payload.get("clarification_count"),
        useful_clarification_count=payload.get("useful_clarification_count"),
        agent_confidence=payload.get("agent_confidence"),
        deterioration_at_classify=payload.get("deterioration_at_classify"),
        additional_info_used=payload.get("additional_info_used"),
        rb_esi_accuracy=reward_breakdown.get("esi_accuracy"),
        rb_reasoning_quality=reward_breakdown.get("reasoning_quality"),
        rb_action_coverage=reward_breakdown.get("action_coverage"),
        rb_temporal_efficiency=reward_breakdown.get("temporal_efficiency"),
        rb_safety_modifier=reward_breakdown.get("safety_modifier"),
        rb_path_quality=reward_breakdown.get("path_quality"),
        rb_final_reward=reward_breakdown.get("final_reward"),
    )


def log_rollout_metrics(summary: Dict[str, Any]) -> None:
    """Emit rollout-batch aggregate metrics for GRPO/PPO style loops."""
    logger = get_logger("medical_triage_env.telemetry")
    logger.info("rollout_batch_metrics", ts=datetime.now(timezone.utc).isoformat(), **summary)
