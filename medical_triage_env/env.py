from __future__ import annotations

from contextlib import asynccontextmanager
import os
import re
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .graders import build_episode_metrics, compute_final_score
from .info_revealer import InfoRevealer
from .logs import get_logger, log_episode_metrics
from .models import (
    ImagingFinding,
    LabResult,
    PatientPresentation,
    TriageAction,
    TriageObservation,
    TriagePhase,
    VitalSigns,
)
from .session import get_session_store
from .tasks import load_all_tasks, TaskConfig

logger = get_logger(__name__)

try:
    _ALL_TASKS, _TASK_IDS = load_all_tasks()
except Exception as _load_err:
    logger.error("task_load_failed", error=str(_load_err))
    _ALL_TASKS, _TASK_IDS = {}, []

# Session store is created inside lifespan so it is never a stale module-level
# singleton when Uvicorn reloads the module (e.g. on HF Spaces restart).
_session_store = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _session_store
    _session_store = get_session_store()
    _session_store.start()
    try:
        yield
    finally:
        _session_store.stop()

app = FastAPI(title="medical-triage-env", version="0.1.0", lifespan=lifespan)


class MedicalTriageEnv:
    def __init__(self, task_id: str, seed: Optional[int] = None, clarify_penalty_dampening: float = 1.0) -> None:
        self.all_tasks, self.task_ids = _ALL_TASKS, _TASK_IDS
        self.clarify_penalty_dampening = clarify_penalty_dampening
        if task_id not in self.all_tasks:
            raise ValueError(
                f"Task {task_id} not found. "
                "Available tasks are intentionally not enumerated."
            )

        self.task_config: TaskConfig = self.all_tasks[task_id]
        self.task_id = task_id
        self.seed = seed
        self.session_id = str(uuid.uuid4())
        self._public_task_ref = f"case-{uuid.uuid4().hex[:8]}"

        self.info_revealer = InfoRevealer(self.task_config, seed=self.seed)

        self.current_step: int = 0
        self.episode_rewards: List[float] = []
        self.action_history: List[TriageAction] = []
        self.done: bool = False
        self.current_vitals: Dict[str, Any] = {}
        self.initial_visible_vitals: Dict[str, Any] = {}
        self._revealed_info: Dict[str, Any] = {}
        self._revealed_labs: List[LabResult] = []
        self._revealed_imaging: List[ImagingFinding] = []
        self._phase_name: str = "assessment"
        self._last_components: Dict[str, float] = {}
        self._last_feedback: str = ""

        # Tracks whether hidden additional_info has been revealed to the agent.
        self._additional_info_revealed: bool = False

        # Avoid leaking the real task_id in logs (task_id is intentionally obfuscated in observations).
        logger.info("env_initialized", session_id=self.session_id)

    def _compute_deterioration_signal(self, vitals: Dict[str, Any]) -> float:
        """Estimate deterioration from visible vitals only (0.0-1.0)."""
        if not self.initial_visible_vitals:
            return 0.0

        # Ensure we only score using clamped/validated vitals.
        try:
            vitals = VitalSigns.model_validate(vitals).model_dump()
        except Exception:
            vitals = dict(vitals or {})

        score = 0.0
        count = 0
        for key, baseline in self.initial_visible_vitals.items():
            if baseline is None or key not in vitals or vitals[key] is None:
                continue
            count += 1
            now = float(vitals[key])
            base = float(baseline)
            if key == "oxygen_saturation":
                # Avoid immediate saturation: a 10-point drop is significant but not always "max".
                delta = max(0.0, (base - now) / 20.0)
            elif key == "blood_pressure_systolic":
                delta = max(0.0, (base - now) / 40.0)
            elif key == "gcs":
                delta = max(0.0, (base - now) / 4.0)
            elif key in {"heart_rate", "respiratory_rate", "temperature"}:
                delta = abs(now - base) / (max(10.0, abs(base) * 0.25))
            else:
                delta = 0.0
            score += min(1.0, delta)

        if count == 0:
            return 0.0
        return round(max(0.0, min(1.0, score / count)), 4)

    def _observation_phase(self) -> TriagePhase:
        if self.done:
            return TriagePhase.COMPLETED
        if self._phase_name in ("disposition", "intervention"):
            # INTERVENTION is planned scaffolding with no implemented action type;
            # surface it as CLASSIFICATION (the disposition-ready phase) to avoid
            # confusing training scripts that branch on phase == ASSESSMENT.
            return TriagePhase.CLASSIFICATION
        return TriagePhase.ASSESSMENT

    def _time_pressure_note(self, deterioration_signal: float) -> Optional[str]:
        steps_left = max(0, self.task_config.max_steps - self.current_step)
        if deterioration_signal >= 0.6 and steps_left <= 2:
            return "Patient deteriorating rapidly - disposition should not be delayed."
        if deterioration_signal >= 0.4:
            return "Clinical trajectory worsening - prioritize critical interventions and triage decision."
        if steps_left <= 1 and not self.done:
            return "Final step remaining - complete triage disposition now."
        return None

    def build_observation(self) -> TriageObservation:
        patient_payload = deepcopy(self.task_config.patient_info.model_dump())
        patient_payload["chief_complaint"] = self.task_config.chief_complaint
        patient_payload["vitals"] = self.current_vitals

        # Hide additional info until the first clarify reveal event.
        if not self._additional_info_revealed:
            patient_payload["additional_info"] = None
        else:
            if self._revealed_info:
                details = [f"{k.replace('_', ' ')}: {v}" for k, v in self._revealed_info.items()]
                info_str = " | ".join(details)
                if len(info_str) > 300:
                    info_str = info_str[:297] + "..."
                patient_payload["additional_info"] = info_str
            else:
                patient_payload["additional_info"] = "Additional history was elicited on clarification."

        patient_payload["revealed_labs"] = [l.model_dump() for l in self._revealed_labs]
        patient_payload["revealed_imaging"] = [i.model_dump() for i in self._revealed_imaging]
        # Confounders are part of the case framing; don't tie their exposure to
        # arbitrary trigger-count ordering.
        patient_payload["confounders_visible"] = list(self.task_config.confounders)

        patient = PatientPresentation.model_validate(patient_payload)
        deterioration_signal = self._compute_deterioration_signal(self.current_vitals)

        return TriageObservation(
            # Opaque per-episode case reference to prevent task-id memorization hacks.
            task_id=self._public_task_ref,
            # 0 on initial reset observation, then increments after each action.
            step_number=min(self.current_step, self.task_config.max_steps),
            max_steps=self.task_config.max_steps,
            phase=self._observation_phase(),
            patient=patient,
            additional_info_revealed=self._additional_info_revealed,
            clarification_history=[
                f"Step {i + 1}: {(a.clarifying_question or 'clarify')[:100]}..."
                for i, a in enumerate(
                    [x for x in self.action_history if x.action_type == "clarify"]
                )
            ][-5:],
            deterioration_signal=deterioration_signal,
            time_pressure_note=self._time_pressure_note(deterioration_signal),
        )

    def reset(self, seed: Optional[int] = None) -> TriageObservation:
        if seed is not None:
            self.seed = seed

        # Only reseed the RNG when an explicit seed is passed to this call.
        # If seed is None we let the RNG advance freely so successive reset()
        # calls produce diverse trajectories (fixes identical-episode bug for RL
        # training when the same env object is reused across episodes).
        self.info_revealer.reset(seed=seed)

        self._public_task_ref = f"case-{uuid.uuid4().hex[:8]}"
        self.current_step = 0
        self.episode_rewards = []
        self.action_history = []
        self.done = False
        self._phase_name = "assessment"
        self._additional_info_revealed = False
        self._revealed_info = {}
        self._revealed_labs = []
        self._revealed_imaging = []
        self._last_components = {}
        self._last_feedback = ""

        self.current_vitals = VitalSigns.model_validate(
            self.info_revealer.get_initial_observation(self.current_step)
        ).model_dump()
        self.initial_visible_vitals = deepcopy(self.current_vitals)

        # Surface any non-hidden initial labs into the observation.
        for lab in self.task_config.initial_labs:
            try:
                if getattr(lab, "hidden", False):
                    continue
                self._revealed_labs.append(
                    LabResult(
                        name=lab.name,
                        value=lab.value,
                        unit=lab.unit,
                        reference_range=lab.reference_range,
                        critical=bool(lab.critical),
                    )
                )
            except Exception:
                continue

        logger.debug("env_reset", session_id=self.session_id)

        return self.build_observation()

    def _record_revealed_artifacts(self, reveal_payload: Dict[str, Any]) -> None:
        revealed = reveal_payload.get("revealed", {})
        if isinstance(revealed, dict):
            self._revealed_info.update(revealed)

        for lab_payload in reveal_payload.get("labs", []):
            try:
                lab = LabResult.model_validate(lab_payload)
            except Exception:
                continue
            if not any(existing.name == lab.name and existing.value == lab.value for existing in self._revealed_labs):
                self._revealed_labs.append(lab)

        for img_payload in reveal_payload.get("imaging", []):
            try:
                img = ImagingFinding.model_validate(img_payload)
            except Exception:
                continue
            if not any(existing.finding == img.finding for existing in self._revealed_imaging):
                self._revealed_imaging.append(img)

        vitals_payload = reveal_payload.get("vitals")
        if not isinstance(vitals_payload, dict) and isinstance(revealed, dict):
            vitals_payload = revealed.get("vitals")
        if isinstance(vitals_payload, dict):
            self.current_vitals.update(vitals_payload)
            # Keep internal state clamped/validated even for reveal-driven updates.
            self.current_vitals = VitalSigns.model_validate(self.current_vitals).model_dump()
        else:
            # Best-effort extraction of structured vitals from common textual patterns
            # revealed by "check_vitals" triggers (e.g. "repeat_bp": "162/94").
            updates: Dict[str, Any] = {}
            if isinstance(revealed, dict):
                bp_text = None
                for k in ("repeat_bp", "bp", "blood_pressure"):
                    if isinstance(revealed.get(k), str):
                        bp_text = revealed.get(k)
                        break
                if isinstance(revealed.get("bp_differential"), str):
                    # Example: "Right arm BP 148/92 — Left arm BP 96/58 — 52mmHg DIFFERENTIAL"
                    bp_text = revealed["bp_differential"]

                if isinstance(bp_text, str):
                    pairs = re.findall(r"(\d{2,3})\s*/\s*(\d{2,3})", bp_text)
                    if pairs:
                        # For differential BP strings (e.g. aortic dissection),
                        # the clinically significant value is the LOWER arm reading
                        # (it indicates obstruction / dissection), not the first
                        # regex match which may be the higher "normal" arm.
                        # Use the minimum systolic across all found pairs.
                        try:
                            sys_val = min(int(s) for s, _d in pairs)
                            # Find the diastolic paired with the chosen systolic.
                            dia_val = next(
                                int(d) for s, d in pairs if int(s) == sys_val
                            )
                            updates["blood_pressure_systolic"] = sys_val
                            updates["blood_pressure_diastolic"] = dia_val
                        except Exception:
                            pass

            if updates:
                self.current_vitals.update(updates)
                self.current_vitals = VitalSigns.model_validate(self.current_vitals).model_dump()

    def _finalize_episode_metrics(self) -> Dict[str, Any]:
        classify_made = any(a.action_type == "classify" for a in self.action_history)

        # Safety blind-spot guard: if the episode ended without any classify
        # action (e.g. clarification-loop exhaustion) and the correct ESI is
        # critical (1–2), record it explicitly so downstream monitoring can
        # surface dangerous stalls rather than silently treating them as
        # "no undertriage" (which they are not — they are unresolved high-risk
        # episodes).
        stalled_on_critical = (
            not classify_made
            and self.task_config.esi_correct <= 2
        )
        if stalled_on_critical:
            logger.warning(
                "stalled_on_critical_case",
                session_id=self.session_id,
                esi_correct=self.task_config.esi_correct,
                steps_taken=self.current_step,
            )

        components = self._last_components or {
            "esi_score": 0.0,
            "temporal_score": 0.0,
            "reasoning_score": 0.0,
            "action_score": 0.0,
            "path_quality": 0.0,
            "safety_modifier": 1.0,
            "final_score": 0.0,
        }
        # Keep episode-level reward telemetry well-defined:
        # - total_reward: terminal step reward (typically in [-1, 1])
        # - extra.episode_return_sum: sum of all step rewards (includes shaping)
        # - extra.shaping_return: sum of non-terminal shaping rewards
        rewards = list(self.episode_rewards)
        episode_return_sum = float(sum(rewards)) if rewards else 0.0
        terminal_reward = float(rewards[-1]) if rewards else 0.0
        shaping_return = float(sum(rewards[:-1])) if len(rewards) > 1 else (0.0 if rewards else 0.0)

        metrics = build_episode_metrics(
            session_id=self.session_id,
            task=self.task_config,
            action_history=self.action_history,
            final_reward=terminal_reward,
            breakdown=components,
            extra_metrics={
                "episode_return_sum": episode_return_sum,
                "shaping_return": shaping_return,
                "classification_made": classify_made,
                "stalled_on_critical": stalled_on_critical,
            },
        )
        log_episode_metrics(metrics)
        payload = metrics.model_dump()
        payload["classification_made"] = classify_made
        payload["ended_without_classification"] = not classify_made
        payload["stalled_on_critical"] = stalled_on_critical
        return payload

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() for a new task.")

        action = TriageAction(
            action_type=(action.action_type or "").strip().lower(),
            esi_level=action.esi_level,
            clarifying_question=action.clarifying_question,
            reasoning=action.reasoning,
            recommended_actions=action.recommended_actions or [],
            confidence=action.confidence,
        )

        self.current_step += 1
        self.action_history.append(action)

        reward = 0.0
        raw_reward = 0.0
        components: Dict[str, float] = {}

        logger.debug(
            "env_step_start",
            session_id=self.session_id,
            step=self.current_step,
            action_type=action.action_type,
            phase=self._phase_name,
        )

        if action.action_type == "clarify":
            clarify_text = action.clarifying_question or "clarify"
            reveal_payload = self.info_revealer.process_clarify(clarify_text, self.current_step)

            if reveal_payload:
                self._additional_info_revealed = True
                self._record_revealed_artifacts(reveal_payload)
                trigger = str(reveal_payload.get("trigger", ""))
                reward = 0.03 if trigger in self.task_config.key_clarify_actions else 0.01
            else:
                reward = -0.01

            clarify_count = sum(1 for a in self.action_history if a.action_type == "clarify")
            if any(tok in clarify_text.lower() for tok in ["ask_history", "check_vitals", "examine_patient"]):
                reward -= 0.02
            if not self.task_config.hidden_info:
                reward -= 0.01
            if clarify_count > self.task_config.expected_clarify_steps + 1:
                reward -= 0.02
            if clarify_count > max(self.task_config.expected_clarify_steps + 2, 4):
                reward -= (0.05 * self.clarify_penalty_dampening)
                if self.clarify_penalty_dampening >= 0.5:
                    self.done = True
                    self._last_feedback = "Clarification loop terminated due to low-value repeated questioning."

            # Multi-phase workflow: assessment -> intervention -> disposition.
            # For tasks with expected_clarify_steps == 0, a clarify action immediately
            # enters disposition (agent has already exceeded expected clarify usage).
            if clarify_count >= max(1, self.task_config.expected_clarify_steps):
                self._phase_name = "disposition"
            elif self._phase_name == "assessment":
                self._phase_name = "intervention"

        elif action.action_type == "classify":
            task_dict = self.task_config.model_dump()

            # Single source of truth for episode-end scoring.
            final_score, components, feedback = compute_final_score(
                action=action,
                task=task_dict,
                action_history=self.action_history,
                steps_taken=self.current_step,
            )
            raw_reward = final_score
            reward = raw_reward
            self.done = True
            self._phase_name = "disposition"
            self._last_components = components
            self._last_feedback = feedback

        else:
            raise ValueError("action_type must be 'classify' or 'clarify'")

        # Never drift terminal states: avoids ghost vitals in final observation.
        # Also avoid drifting after the last allowed step (which becomes terminal via max_steps).
        if (not self.done) and (self.current_step < self.task_config.max_steps):
            drifted = self.info_revealer.apply_vital_drift(
                self.current_vitals,
                self.current_step,
            )
            # Clamp/validate after drift so internal state stays within bounds.
            self.current_vitals = VitalSigns.model_validate(drifted).model_dump()

        if self.current_step >= self.task_config.max_steps:
            if not self.done:
                # Apply an actual timeout penalty even if this was a clarify-only episode.
                reward = max(-1.0, reward - 0.10)
                self._last_feedback = "Episode reached max steps before disposition."
            self.done = True
            self._phase_name = "disposition"

        reward = float(reward)
        self.episode_rewards.append(reward)
        if self.done:
            self._phase_name = "completed"
        next_obs = self.build_observation()

        logger.debug(
            "env_step_end",
            session_id=self.session_id,
            step=self.current_step,
            reward=reward,
            done=self.done,
            cumulative=round(sum(self.episode_rewards), 2),
            phase=self._phase_name,
        )

        info = {
            "raw_score": raw_reward if action.action_type == "classify" else reward,
            "step": self.current_step,
            "grader_feedback": self._last_feedback,
            "additional_info_revealed": self._additional_info_revealed,
            "score_components": components,
            "workflow_phase": self._phase_name,
            "revealed_trigger_count": len(self.info_revealer.get_revealed_info_keys()),
            "task_ref": self._public_task_ref,
        }

        if self.done:
            info["episode_metrics"] = self._finalize_episode_metrics()
            info["classification_made"] = any(a.action_type == "classify" for a in self.action_history)

        return next_obs, reward, self.done, info

    def state(self) -> dict:
        return {
            "session_id": self.session_id,
            # Do not leak internal task ids; use the per-episode opaque reference.
            "task_ref": self._public_task_ref,
            "step": self.current_step,
            "done": self.done,
            "episode_rewards": list(self.episode_rewards),
            "cumulative_score": round(sum(self.episode_rewards), 2),
            "additional_info_revealed": self._additional_info_revealed,
            "workflow_phase": self._phase_name,
            "revealed_triggers": self.info_revealer.get_revealed_info_keys(),
            "revealed_labs": [l.model_dump() for l in self._revealed_labs],
            "revealed_imaging": [i.model_dump() for i in self._revealed_imaging],
            "action_history_summary": [
                {
                    "step": i + 1,
                    "action_type": a.action_type,
                    "esi_level": a.esi_level,
                    "confidence": a.confidence,
                }
                for i, a in enumerate(self.action_history)
            ],
            "max_steps": self.task_config.max_steps,
        }




def _get_store():
    """Return the active session store, or raise 503 if lifespan hasn't fired yet.

    This guards against test harnesses or HF Spaces health probes that hit
    an endpoint before Uvicorn's lifespan context has initialised the store.
    In production Uvicorn, lifespan always completes before the first request.
    """
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up, please retry.")
    return _session_store


@app.post("/reset")
def reset_endpoint(payload: Optional[dict] = Body(default=None)) -> Dict[str, Any]:
    # Validate task_id without leaking internal task corpus in error messages.
    task_id: str
    if payload and "task_id" in payload and payload["task_id"] is not None:
        if not isinstance(payload["task_id"], str):
            raise HTTPException(status_code=400, detail="task_id must be a string")
        task_id = payload["task_id"].strip()
        if not task_id or task_id not in _ALL_TASKS:
            raise HTTPException(status_code=404, detail="Task not found")
    else:
        task_ids = _TASK_IDS
        if not task_ids:
            raise HTTPException(status_code=500, detail="No tasks available")
        task_id = task_ids[0]

    seed: Optional[int] = None
    if payload and "seed" in payload and payload["seed"] is not None:
        try:
            seed = int(payload["seed"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="seed must be an integer") from exc
    
    try:
        env = MedicalTriageEnv(task_id, seed=seed)
        # __init__ already seeded the InfoRevealer via seed=seed; call reset()
        # without a seed so the RNG is not redundantly re-applied to the same
        # value (which would cause every reset() call to replay the same
        # trajectory — see reset() docstring for details).
        observation = env.reset()
        _get_store().create(env)

        # Do not leak internal task ids in logs.
        logger.info("environment_reset_success", session_id=env.session_id)

        return {
            "observation": observation.model_dump(),
            "info": {
                "session_id": env.session_id,
                # Expose only the opaque per-episode reference.
                "task_ref": env._public_task_ref,
                "seed": seed,
            },
        }
        
    except ValueError:
        # Avoid leaking internal task ids or other sensitive details.
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as exc:
        logger.error("environment_reset_error", error=str(exc), task_id=task_id)
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {exc}") from exc


@app.post("/step")
def step_endpoint(payload: dict = Body(...)) -> Dict[str, Any]:
    if "session_id" not in payload:
        raise HTTPException(status_code=400, detail="session_id is required")
    if "action" not in payload:
        raise HTTPException(status_code=400, detail="action is required")

    session_id = payload["session_id"]
    env = _get_store().get(session_id)

    try:
        action = TriageAction.model_validate(payload["action"])
    except Exception:
        # Avoid leaking pydantic schema structure in error responses.
        raise HTTPException(status_code=400, detail="Invalid action payload.")

    try:
        observation, reward, done, info = env.step(action)
        _get_store().save(env)

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        # Client sent semantically invalid data (e.g. bad action_type).
        logger.warning("step_client_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Genuine server-side failure — do not pretend it is a client error.
        logger.error("step_execution_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error during step.") from exc


@app.get("/state")
def state_endpoint(session_id: str):
    # Pure read: get() already bumps last-access TTL; an extra save() here
    # would add needless write overhead and raise 404 if the session expired
    # between get() and save() — confusing for a read-only endpoint.
    env = _get_store().get(session_id)
    return env.state()


@app.get("/tasks")
def tasks_endpoint():
    try:
        # Avoid exposing internal task ids publicly by default.
        # For local development debugging, set ENV=development.
        deploy_env = os.getenv("ENV", "production").lower()
        task_ids = _TASK_IDS
        if deploy_env == "development":
            return {"tasks": task_ids, "count": len(task_ids)}
        return {"tasks": [], "count": len(task_ids)}
    except Exception as exc:
        logger.error("tasks_list_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_endpoint() -> dict:
    store = _get_store()
    return {
        "status": "ok",
        "active_sessions": len(store.list_active()),
    }


@app.get("/")
def root_endpoint() -> HTMLResponse:
        return HTMLResponse(content="""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Medical Triage Environment</title>
</head>
<body>
    <h1>Medical Triage Environment</h1>
    <ul>
        <li><code>GET  /health</code></li>
        <li><code>POST /reset</code></li>
        <li><code>POST /step</code></li>
        <li><code>GET  /state?session_id=...</code></li>
        <li><code>GET  /tasks</code></li>
    </ul>
</body>
</html>""")


@app.get("/web")
def web_endpoint() -> HTMLResponse:
    return root_endpoint()
