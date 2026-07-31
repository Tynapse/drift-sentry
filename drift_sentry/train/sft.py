from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from drift_sentry.data.conversations import load_conversation_rows, prompt_messages, rows_to_conversations
from drift_sentry.train.runtime_config import build_training_run_summary, load_training_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT/LoRA training entrypoint for DriftSentry datasets.")
    parser.add_argument("--config", required=True, help="YAML training config.")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps.")
    parser.add_argument("--num-samples", type=int, help="Limit training samples for smoke runs.")
    parser.add_argument("--model", help="Override model/base_model.")
    parser.add_argument("--output-dir", help="Override training.output_dir.")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--max-seq-length", type=int, help="Override data.max_seq_length.")
    parser.add_argument("--epochs", type=float, help="Override training.epochs.")
    parser.add_argument("--lora-r", type=int, help="Override lora.r.")
    parser.add_argument("--lora-alpha", type=int, help="Override lora.alpha.")
    parser.add_argument("--lora-dropout", type=float, help="Override lora.dropout.")
    parser.add_argument("--resume-adapter", help="Load an existing PEFT adapter and continue training it.")
    parser.add_argument("--full-finetuning", action="store_true", help="Train all model weights instead of LoRA.")
    parser.add_argument("--skip-merge", action="store_true", help="Do not merge the LoRA adapter after training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the resolved run summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = _build_overrides(args)

    if args.dry_run:
        summary = build_training_run_summary(
            workflow="sft",
            config_path=args.config,
            overrides=overrides,
            dry_run=True,
        )
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    try:
        return run_training(args, overrides)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    if args.num_samples is not None:
        overrides.setdefault("data", {})["num_samples"] = args.num_samples
    if args.output_dir:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        overrides.setdefault("training", {})["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_seq_length is not None:
        overrides.setdefault("data", {})["max_seq_length"] = args.max_seq_length
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.model:
        overrides["model"] = {"base_model": args.model}
    if args.resume_adapter:
        overrides.setdefault("model", {})["adapter_path"] = args.resume_adapter
    if args.lora_r is not None:
        overrides.setdefault("lora", {})["r"] = args.lora_r
    if args.lora_alpha is not None:
        overrides.setdefault("lora", {})["alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        overrides.setdefault("lora", {})["dropout"] = args.lora_dropout
    if args.full_finetuning:
        overrides.setdefault("training", {})["full_finetuning"] = True
    return overrides


def run_training(args: argparse.Namespace, overrides: dict[str, Any]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config)
    config = load_training_config(config_path, overrides=overrides)
    data_cfg = _section(config.get("data"))
    training_cfg = _section(config.get("training") or config.get("train"))
    model_cfg = _section(config.get("model"))
    lora_cfg = _section(config.get("lora") or training_cfg.get("lora"))

    base_model = _model_name(config)
    if not base_model:
        raise ValueError("model/base_model is required")
    model_revision = _model_revision(model_cfg)

    train_rows = _load_train_rows(data_cfg)
    eval_rows = _load_eval_rows(data_cfg)
    if not train_rows:
        raise ValueError("training data is empty")

    output_dir = _resolve_output_dir(config_path, config, training_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded DriftSentry SFT rows: train=%s eval=%s", len(train_rows), len(eval_rows))
    logger.info("Output directory: %s", output_dir)

    try:
        import torch  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-untyped]
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("install tynapse-drift-sentry[train] to run SFT training") from exc

    from drift_sentry.train.lora import create_lora_config

    _set_local_cuda_device(torch)

    seed = int(training_cfg.get("seed", config.get("seed", 42)))
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=model_revision,
        trust_remote_code=_trust_remote_code(model_cfg),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_seq_length = int(data_cfg.get("max_seq_length", training_cfg.get("max_seq_length", 4096)))
    train_conversations = rows_to_conversations(train_rows)
    eval_conversations = rows_to_conversations(eval_rows) if eval_rows else []

    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        full_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        prompt_text = _apply_chat_template(tokenizer, prompt_messages(messages), add_generation_prompt=True)
        full = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False)
        prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, padding=False)
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        for index in range(prompt_len):
            labels[index] = -100
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "mm_token_type_ids": [0] * len(full["input_ids"]),
            "labels": labels,
            "trainable_tokens": sum(1 for label in labels if label != -100),
            "source_id": row.get("id", ""),
        }

    train_ds = Dataset.from_list(train_conversations).map(
        tokenize_row,
        remove_columns=list(train_conversations[0].keys()),
    )
    train_ds = train_ds.filter(lambda row: int(row["trainable_tokens"]) > 0)
    if len(train_ds) == 0:
        raise ValueError("all training rows have zero assistant tokens after truncation")

    eval_ds = None
    if eval_conversations:
        eval_ds = Dataset.from_list(eval_conversations).map(
            tokenize_row,
            remove_columns=list(eval_conversations[0].keys()),
        )
        eval_ds = eval_ds.filter(lambda row: int(row["trainable_tokens"]) > 0)
        if len(eval_ds) == 0:
            eval_ds = None

    class CausalCollator:
        def __init__(self, tokenizer_obj: Any) -> None:
            self._padder = DataCollatorWithPadding(tokenizer=tokenizer_obj, padding=True)

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            labels = [feature["labels"] for feature in features]
            mm_token_type_ids = [feature["mm_token_type_ids"] for feature in features]
            model_features = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ]
            batch = self._padder(model_features)
            max_len = batch["input_ids"].shape[1]
            padded_labels = [label + [-100] * (max_len - len(label)) for label in labels]
            padded_mm_token_type_ids = [ids + [0] * (max_len - len(ids)) for ids in mm_token_type_ids]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            batch["mm_token_type_ids"] = torch.tensor(padded_mm_token_type_ids, dtype=torch.long)
            return batch

    load_in_4bit = bool(model_cfg.get("load_in_4bit", training_cfg.get("load_in_4bit", False)))
    full_finetuning = bool(training_cfg.get("full_finetuning", False))
    bf16 = bool(model_cfg.get("bf16", training_cfg.get("bf16", torch.cuda.is_available())))
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=model_revision,
        trust_remote_code=_trust_remote_code(model_cfg),
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=_device_map(model_cfg),
    )
    _unwrap_gemma4_clippable_linears(model)
    if bool(training_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    adapter_path = _adapter_path(model_cfg, training_cfg)
    if not full_finetuning and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        logger.info("Loaded trainable PEFT adapter: %s", adapter_path)
    elif not full_finetuning:
        model = get_peft_model(model, create_lora_config(lora_cfg))
        model.print_trainable_parameters()
    else:
        _log_trainable_parameters(model)

    has_eval = eval_ds is not None
    report_to = training_cfg.get("report_to", "none")
    report_to_value: list[str] | str
    report_to_value = [] if report_to == "none" else str(report_to)
    max_steps = int(training_cfg.get("max_steps", -1))
    distributed_kwargs: dict[str, Any] = {}
    if training_cfg.get("fsdp"):
        distributed_kwargs["fsdp"] = str(training_cfg["fsdp"])
    if isinstance(training_cfg.get("fsdp_config"), dict):
        distributed_kwargs["fsdp_config"] = training_cfg["fsdp_config"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        num_train_epochs=float(training_cfg.get("epochs", 1)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", training_cfg.get("per_device_train_batch_size", 1))),
        per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 1))),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        bf16=bf16 and torch.cuda.is_available(),
        fp16=not bf16 and torch.cuda.is_available(),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(training_cfg.get("eval_steps", training_cfg.get("logging_steps", 50))),
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_steps", 100)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=has_eval and bool(training_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=report_to_value,
        seed=seed,
        optim=str(training_cfg.get("optim", "adamw_torch")),
        ddp_find_unused_parameters=bool(training_cfg.get("ddp_find_unused_parameters", not full_finetuning)),
        **distributed_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_config_resolved.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "model": base_model,
                "model_revision": model_revision,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "tokenized_train_rows": len(train_ds),
                "tokenized_eval_rows": len(eval_ds) if eval_ds is not None else 0,
                "full_finetuning": full_finetuning,
                "config": config,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    merge = bool(training_cfg.get("merge", False)) and not args.skip_merge and not full_finetuning
    if merge:
        merged_dir = output_dir / "merged"
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=model_revision,
            trust_remote_code=_trust_remote_code(model_cfg),
            torch_dtype=dtype,
            device_map=_device_map(model_cfg),
        )
        _unwrap_gemma4_clippable_linears(base)
        merged = PeftModel.from_pretrained(base, str(output_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("Merged model saved: %s", merged_dir)

    logger.info("SFT training complete: %s", output_dir)
    return 0


def _load_train_rows(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return load_conversation_rows(
        repo_id=_string_or_none(data_cfg.get("repo_id")),
        config_name=_string_or_none(data_cfg.get("config_name") or data_cfg.get("config")),
        split=str(data_cfg.get("split", data_cfg.get("train_split", "train"))),
        data_path=data_cfg.get("train_file") or data_cfg.get("train_path"),
        limit=_int_or_none(data_cfg.get("num_samples")),
    )


def _load_eval_rows(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    eval_path = data_cfg.get("validation_file") or data_cfg.get("val_file") or data_cfg.get("eval_file")
    eval_config = data_cfg.get("validation_config_name") or data_cfg.get("eval_config_name") or data_cfg.get("eval_config")
    eval_split = data_cfg.get("validation_split") or data_cfg.get("eval_split")
    if not eval_path and not eval_config and not eval_split:
        return []
    return load_conversation_rows(
        repo_id=_string_or_none(data_cfg.get("eval_repo_id") or data_cfg.get("repo_id")),
        config_name=_string_or_none(eval_config),
        split=str(eval_split or "validation"),
        data_path=eval_path,
        limit=_int_or_none(data_cfg.get("eval_num_samples")),
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return _fallback_chat_template(messages, add_generation_prompt=add_generation_prompt)
    except Exception:
        return _fallback_chat_template(messages, add_generation_prompt=add_generation_prompt)


def _fallback_chat_template(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    parts = [f"<|{message['role']}|>\n{message['content']}" for message in messages]
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _resolve_output_dir(config_path: Path, config: dict[str, Any], training_cfg: dict[str, Any]) -> Path:
    _ = config_path
    output = training_cfg.get("output_dir") or config.get("output_dir") or "artifacts/sft"
    output_dir = Path(str(output))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if bool(training_cfg.get("timestamp_output_dir", False)):
        exp_name = str(config.get("name") or config.get("exp_name") or "sft")
        output_dir = output_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir


def _model_name(config: dict[str, Any]) -> str:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("base_model") or model_cfg.get("name") or "")
    return str(model_cfg or config.get("base_model") or "")


def _adapter_path(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> str:
    return str(model_cfg.get("adapter_path") or training_cfg.get("adapter_path") or "").strip()


def _model_revision(model_cfg: dict[str, Any]) -> str | None:
    value = str(model_cfg.get("revision") or "").strip()
    return value or None


def _trust_remote_code(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("trust_remote_code", True))


def _device_map(model_cfg: dict[str, Any]) -> Any:
    """Avoid accidental model sharding when launched through torchrun/DDP."""

    if int(os.environ.get("WORLD_SIZE") or "1") > 1:
        return {"": int(os.environ.get("LOCAL_RANK") or "0")}
    return model_cfg.get("device_map", "auto")


def _set_local_cuda_device(torch_module: Any) -> None:
    if int(os.environ.get("WORLD_SIZE") or "1") <= 1:
        return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.set_device(int(os.environ.get("LOCAL_RANK") or "0"))


def _unwrap_gemma4_clippable_linears(model: Any) -> None:
    for name, child in list(model.named_children()):
        if type(child).__name__ == "Gemma4ClippableLinear" and hasattr(child, "linear"):
            setattr(model, name, child.linear)
        else:
            _unwrap_gemma4_clippable_linears(child)


def _log_trainable_parameters(model: Any) -> None:
    trainable = total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        trainable += count if parameter.requires_grad else 0
    ratio = 100 * trainable / total if total else 0.0
    logger.info("trainable params: %s || all params: %s || trainable%%: %.4f", trainable, total, ratio)


def _section(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
