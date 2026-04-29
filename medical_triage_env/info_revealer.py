"""Information reveals and stochastic deterioration for the triage environment."""
from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Dict, List, Optional, Set

from triagerl.core.constants import KEYWORD_TO_TRIGGER, VALID_TRIGGERS
from triagerl.tasks.schema import TaskConfig
from triagerl.logs.logger import get_logger

logger = get_logger(__name__)


class InfoRevealer:
    """Manages progressive information disclosure and stochastic vital drift."""

    # Internal trigger token names — blocked in raw questions to prevent
    # reward hacking via direct trigger injection.
    # Sourced from constants.VALID_TRIGGERS (same set, exposed as class attr).
    DIRECT_TRIGGER_TOKENS = VALID_TRIGGERS

    # Keyword → trigger mapping — canonical copy lives in constants.py.
    # Any updates must be made there.
    KEYWORD_TO_TRIGGER = KEYWORD_TO_TRIGGER

    def __init__(
        self,
        task: TaskConfig,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.task = task
        self.revealed_triggers: Set[str] = set()
        self.seed = seed
        self.rng = rng or random.Random(seed)

        logger.debug(
            "info_revealer_initialized",
            task_id=task.id,
            hidden_info_count=len(task.hidden_info),
            vital_drift_enabled=bool(task.vital_drift.per_step),
            seed=seed,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset per-episode reveal state and optionally reseed stochastic drift.

        Args:
            seed: When provided, updates ``self.seed`` and reseeds the RNG to
                  that value — use this for deterministic single-episode replay.
                  When ``None`` (the default), the RNG is **not** reseeded and
                  continues from its current state, giving diverse trajectories
                  across successive episodes on the same object.

        .. note::
            Do **not** pass the constructor seed on every call if you want
            trajectory diversity.  Doing so resets the RNG to the same starting
            state and every episode will produce an identical drift sequence.
        """
        self.revealed_triggers.clear()
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        # No elif: when seed is None we deliberately leave the RNG advancing
        # so that repeated reset() calls without an explicit seed produce
        # statistically independent episodes.

    def get_initial_observation(self, step: int) -> Dict[str, Any]:
        """Get initial visible vitals, hiding keys tagged as hidden_vitals."""
        vitals = deepcopy(self.task.initial_vitals)

        hidden_keys = set()
        for hidden_item in self.task.hidden_info:
            data = hidden_item.data
            if "hidden_vitals" in data:
                hidden_keys.update(data["hidden_vitals"])

        for key in hidden_keys:
            vitals.pop(key, None)

        logger.debug(
            "initial_observation_created",
            task_id=self.task.id,
            step=step,
            visible_vitals=list(vitals.keys()),
            hidden_vitals=list(hidden_keys),
        )

        return vitals

    def infer_trigger(self, clarifying_text: str) -> str:
        """Infer best trigger from free-text clarification query."""
        text = (clarifying_text or "").lower().strip()
        if not text:
            return "clarify"

        # Prevent reward hacking via direct internal trigger-token injection.
        if any(token in text for token in self.DIRECT_TRIGGER_TOKENS):
            return "clarify"

        # Aggregate keyword evidence so behavior is deterministic and independent
        # of dictionary insertion order when multiple concepts are mentioned.
        trigger_scores: Dict[str, int] = {}
        for keyword, trigger in self.KEYWORD_TO_TRIGGER.items():
            if keyword in text:
                trigger_scores[trigger] = trigger_scores.get(trigger, 0) + len(keyword)

        if trigger_scores:
            return max(sorted(trigger_scores), key=lambda t: trigger_scores[t])
        return "clarify"

    def _is_meaningful_clarification(self, clarifying_text: str) -> bool:
        """Gate reveals so short or token-hack prompts do not unlock hidden layers."""
        text = (clarifying_text or "").strip().lower()
        if len(text.split()) < 4:
            return False
        if any(token in text for token in self.DIRECT_TRIGGER_TOKENS):
            return False
        # Encourage natural clinical questions over template spam.
        if "?" in text:
            return True
        return any(w in text for w in ["what", "when", "how", "any", "history", "exam", "vitals", "changes"])

    def process_clarify(self, clarifying_text: str, step: int) -> Dict[str, Any]:
        """Apply layered reveal logic for a single clarify action."""
        if not self._is_meaningful_clarification(clarifying_text):
            logger.debug(
                "clarify_rejected_low_quality",
                task_id=self.task.id,
                step=step,
            )
            return {}

        trigger = self.infer_trigger(clarifying_text)
        if trigger in self.revealed_triggers:
            return {}

        revealed_layers: List[Dict[str, Any]] = []
        for hidden_item in self.task.hidden_info:
            if hidden_item.trigger == trigger:
                revealed_layers.append(hidden_item.data)

        if not revealed_layers:
            return {}

        self.revealed_triggers.add(trigger)

        merged: Dict[str, Any] = {
            "trigger": trigger,
            "revealed": {},
            "labs": [],
            "imaging": [],
        }

        for layer in revealed_layers:
            for k, v in layer.items():
                merged["revealed"][k] = v
                key_low = k.lower()
                text = str(v)
                if any(tok in key_low for tok in ["ecg", "ct", "xray", "x-ray", "mri", "ultrasound", "echo", "cxr"]):
                    merged["imaging"].append({"modality": "clinical finding", "finding": f"{k}: {text}", "critical": False})
                if any(tok in key_low for tok in ["inr", "lactate", "troponin", "bnp", "wcc", "crp", "creatinine", "anti-xa", "glucose"]):
                    merged["labs"].append(
                        {
                            "name": k,
                            "value": text,
                            "unit": "",
                            "reference_range": "",
                            "critical": any(flag in text.lower() for flag in ["critical", "high", "low", "supratherapeutic"]),
                        }
                    )

        logger.debug(
            "info_revealed",
            task_id=self.task.id,
            trigger=trigger,
            step=step,
            reveal_keys=list(merged["revealed"].keys()),
            labs_revealed=len(merged["labs"]),
            imaging_revealed=len(merged["imaging"]),
        )
        return merged

    def apply_vital_drift(self, vitals: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Apply stochastic per-step drift after configured start step."""
        if step < self.task.vital_drift.starts_at_step:
            return vitals

        if not self.task.vital_drift.per_step:
            return vitals

        drifted_vitals = deepcopy(vitals)

        for vital_key, drift_per_step in self.task.vital_drift.per_step.items():
            if vital_key in drifted_vitals and drifted_vitals[vital_key] is not None:
                current_value = float(drifted_vitals[vital_key])
                sigma = float(self.task.vital_drift.noise_sigma.get(vital_key, 0.0))
                noise = self.rng.gauss(0.0, sigma) if sigma > 0 else 0.0
                drifted_value = current_value + drift_per_step + noise

                if vital_key == "heart_rate":
                    drifted_value = max(20.0, min(300.0, drifted_value))
                elif vital_key == "oxygen_saturation":
                    drifted_value = max(50.0, min(100.0, drifted_value))
                elif vital_key == "blood_pressure_systolic":
                    drifted_value = max(40.0, min(300.0, drifted_value))
                elif vital_key == "respiratory_rate":
                    drifted_value = max(4.0, min(60.0, drifted_value))
                elif vital_key == "gcs":
                    drifted_value = max(3.0, min(15.0, drifted_value))

                if vital_key in ["temperature", "oxygen_saturation"]:
                    drifted_vitals[vital_key] = round(drifted_value, 1)
                else:
                    drifted_vitals[vital_key] = int(round(drifted_value))

                logger.debug(
                    "vital_drift_applied",
                    task_id=self.task.id,
                    vital=vital_key,
                    original=current_value,
                    drifted=drifted_vitals[vital_key],
                    step=step,
                    drift_per_step=drift_per_step,
                    noise_sigma=sigma,
                    sampled_noise=round(noise, 4),
                )

        return drifted_vitals

    def get_confounders(self) -> List[str]:
        """Get confounders visible in this task."""
        return list(self.task.confounders)

    def get_revealed_info_keys(self) -> List[str]:
        """Get reveal triggers already unlocked."""
        return list(self.revealed_triggers)
