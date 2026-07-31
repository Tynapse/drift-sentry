from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drift_sentry.utils.config import load_yaml
from drift_sentry.utils.merge import deep_merge


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    workflow: str
    config_path: Path
    experiment_name: str | None
    model: str | None
    dataset: str | None
    output_dir: str | None
    max_steps: int | None
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "config_path": str(self.config_path),
            "experiment_name": self.experiment_name,
            "model": self.model,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "max_steps": self.max_steps,
            "dry_run": self.dry_run,
        }


def load_training_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_yaml(path)
    if overrides:
        config = deep_merge(config, overrides)
    return config


def build_training_run_summary(
    *,
    workflow: str,
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> TrainingRunSummary:
    resolved_path = Path(config_path)
    config = load_training_config(resolved_path, overrides=overrides)
    training = _mapping_or_empty(config.get("training"))
    data = _mapping_or_empty(config.get("data"))
    model = _mapping_or_empty(config.get("model"))

    return TrainingRunSummary(
        workflow=workflow,
        config_path=resolved_path,
        experiment_name=_string(config.get("exp_name") or config.get("name")),
        model=_string(model.get("base_model") or config.get("model") or config.get("base_model") or config.get("teacher_model")),
        dataset=_string(data.get("dataset") or data.get("train_file") or data.get("config_name") or config.get("dataset")),
        output_dir=_string(config.get("output_dir") or training.get("output_dir")),
        max_steps=_int_or_none(training.get("max_steps") or config.get("max_steps")),
        dry_run=dry_run,
    )


def require_training_runtime(workflow: str) -> None:
    msg = (
        f"{workflow} full training runtime is intentionally gated behind the tynapse-drift-sentry[train] extra "
        "and a GPU host. This migration entrypoint currently validates config/CLI shape locally; run the "
        "TYN-1 GPU validation before enabling it as an active production training path."
    )
    raise RuntimeError(msg)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
