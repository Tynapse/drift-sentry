from __future__ import annotations

from typing import Any

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def create_lora_config(config: dict[str, Any]) -> Any:
    """Create a PEFT LoraConfig from a trust-layer training config fragment.

    The PEFT dependency lives in the optional `train` extra, so importing this
    module remains safe in minimal data-only environments.
    """
    try:
        from peft import LoraConfig  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("peft is required for create_lora_config; install tynapse-drift-sentry[train]") from exc
    target_modules = config.get("target_modules", DEFAULT_TARGET_MODULES)
    lora_target_modules: str | list[Any]
    if isinstance(target_modules, str):
        lora_target_modules = target_modules
    else:
        lora_target_modules = list(target_modules)
    return LoraConfig(
        r=int(config.get("r", 128)),
        lora_alpha=int(config.get("alpha", 256)),
        lora_dropout=float(config.get("dropout", 0.05)),
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )
