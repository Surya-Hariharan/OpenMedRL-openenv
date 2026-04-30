from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any, Mapping


_LOGGER_LOCK = Lock()


def _format_value(value: Any) -> str:
	if isinstance(value, str):
		return value
	try:
		return json.dumps(value, sort_keys=True, default=str)
	except TypeError:
		return str(value)


def _format_event(event: str, fields: Mapping[str, Any]) -> str:
	if not fields:
		return event
	payload = " ".join(f"{key}={_format_value(fields[key])}" for key in sorted(fields))
	return f"{event} {payload}"


def _configure_logger(logger: logging.Logger) -> None:
	if getattr(logger, "_triagerl_configured", False):
		return

	with _LOGGER_LOCK:
		if getattr(logger, "_triagerl_configured", False):
			return

		if not logger.handlers:
			handler = logging.StreamHandler()
			handler.setFormatter(
				logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
			)
			logger.addHandler(handler)

		logger.setLevel(logging.INFO)
		logger.propagate = False
		logger._triagerl_configured = True


class StructuredLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, event: str, /, **fields: Any) -> None:
        self._logger.debug(_format_event(event, fields))

    def info(self, event: str, /, **fields: Any) -> None:
        self._logger.info(_format_event(event, fields))

    def warning(self, event: str, /, **fields: Any) -> None:
        self._logger.warning(_format_event(event, fields))

    def error(self, event: str, /, **fields: Any) -> None:
        self._logger.error(_format_event(event, fields))

    def critical(self, event: str, /, **fields: Any) -> None:
        self._logger.critical(_format_event(event, fields))

    def exception(self, event: str, /, **fields: Any) -> None:
        self._logger.exception(_format_event(event, fields))


def get_logger(name: str) -> StructuredLogger:
    logger = logging.getLogger(name)
    _configure_logger(logger)
    return StructuredLogger(logger)


def log_episode_metrics(logger: StructuredLogger, metrics: dict[str, Any]) -> None:
    logger.info("episode_metrics", **metrics)


def log_rollout_metrics(logger: StructuredLogger, metrics: dict[str, Any]) -> None:
    logger.info("rollout_metrics", **metrics)
