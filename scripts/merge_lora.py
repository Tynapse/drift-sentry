#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a self-contained Hugging Face model.

The Qwen3.5 multimodal wrapper requires special handling so the merged output
keeps the architecture expected by vLLM. Adapter-key prefixes are normalized
before PEFT merge to avoid silently producing base-only weights.

Usage:
  python3 merge_lora.py \\
      --base    /path/to/qwen3.5-2b-base \\
      --adapter /path/to/input_guard_v6 \\
      --output  /path/to/merged_dir
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    for required in (args.base, args.adapter):
        if not (required / "config.json").exists() and \
           not (required / "adapter_config.json").exists():
            print(f"FATAL: {required} missing config.json/adapter_config.json",
                  file=sys.stderr)
            return 2

    import json
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer)

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # Multimodal bases (Qwen3.5-2B-Base ships with video_preprocessor_config.json)
    # must be loaded via AutoModelForImageTextToText so save_pretrained preserves
    # the multimodal wrapper (model_type="qwen3_5", architectures includes
    # ConditionalGeneration). AutoModelForCausalLM would unwrap to the language
    # submodule and save with model_type="qwen3_5_text", which vLLM's Qwen3_5
    # model class rejects with `Invalid type of HuggingFace config`.
    is_multimodal = (args.base / "video_preprocessor_config.json").exists() or \
                    (args.base / "preprocessor_config.json").exists()
    if is_multimodal:
        print(f"[merge] loading base (multimodal — image-text-to-text): {args.base}",
              file=sys.stderr)
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.base), torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa", device_map="cpu",
        )
    else:
        print(f"[merge] loading base (text-only causal LM): {args.base}",
              file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.base), torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa", device_map="cpu",
        )
    # The adapter was trained against a text-only view of the base — its
    # safetensors keys look like `base_model.model.model.layers.0.mlp.…`.
    # In the full multimodal model, those modules live at
    # `language_model.model.layers.0.mlp.…` (nested under the VLM wrapper).
    # PEFT matches by full key, so without a rename it silently merges nothing
    # and the saved model still produces base-only outputs. Rewrite the keys
    # to add whatever prefix is needed before loading.
    print(f"[merge] applying adapter: {args.adapter}", file=sys.stderr)
    if is_multimodal:
        # Discover the prefix by searching for one of the trained target modules
        # in the loaded model. Look for a module path ending with
        # `model.layers.0.mlp.down_proj` (or similar); strip that suffix to get
        # the prefix we need to insert into adapter keys.
        wanted_suffix = "model.layers.0.mlp.down_proj"
        prefix = None
        for name, _ in model.named_modules():
            if name.endswith(wanted_suffix):
                prefix = name[: -len(wanted_suffix)]  # e.g. "language_model."
                break
        if prefix is None:
            raise RuntimeError(
                f"could not locate '{wanted_suffix}' in multimodal model — "
                "named_modules() prefix unknown")
        if prefix:
            print(f"[merge]   detected language-model prefix: '{prefix}'",
                  file=sys.stderr)
            from safetensors import safe_open
            from safetensors.torch import save_file
            tmp_adapter = args.output.parent / "_adapter_renamed"
            tmp_adapter.mkdir(parents=True, exist_ok=True)
            new_state = {}
            with safe_open(str(args.adapter / "adapter_model.safetensors"),
                           framework="pt") as f:
                for k in f.keys():
                    new_k = k
                    # Rewrite "base_model.model.<rest>" → "base_model.model.<prefix><rest>"
                    head = "base_model.model."
                    if k.startswith(head):
                        new_k = head + prefix + k[len(head):]
                    new_state[new_k] = f.get_tensor(k)
            save_file(new_state, str(tmp_adapter / "adapter_model.safetensors"))
            shutil.copy2(args.adapter / "adapter_config.json",
                         tmp_adapter / "adapter_config.json")
            adapter_to_load = tmp_adapter
        else:
            adapter_to_load = args.adapter
    else:
        adapter_to_load = args.adapter

    model = PeftModel.from_pretrained(model, str(adapter_to_load))
    # Sanity: confirm at least one LoRA module was actually attached.
    n_lora = sum(1 for n, _ in model.named_modules() if "lora_A" in n)
    print(f"[merge]   PEFT attached {n_lora} lora_A modules", file=sys.stderr)
    if n_lora == 0:
        raise RuntimeError(
            "PEFT did not attach any LoRA modules — adapter key prefix is wrong")
    print("[merge] running merge_and_unload()", file=sys.stderr)
    model = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged model to: {args.output}", file=sys.stderr)
    model.save_pretrained(str(args.output), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(args.base), trust_remote_code=True)
    tok.save_pretrained(str(args.output))

    # Carry over preprocessor configs (multimodal Qwen3.5 base ships these and
    # vLLM expects them alongside config.json — it errors otherwise).
    for extra in ("preprocessor_config.json", "video_preprocessor_config.json",
                  "chat_template.jinja", "generation_config.json"):
        src = args.base / extra
        if src.exists():
            shutil.copy2(src, args.output / extra)
            print(f"[merge]   carried over {extra}", file=sys.stderr)

    # Defensive: if save_pretrained still wrote a text-only config (some
    # transformers versions strip the multimodal wrapper when AutoModel
    # resolves to the language submodule), overlay the original base config
    # so vLLM's Qwen3_5Config type-check passes. The actual weights are
    # already merged in the safetensors.
    out_config = args.output / "config.json"
    base_config = args.base / "config.json"
    if out_config.exists() and base_config.exists():
        try:
            mt = json.loads(out_config.read_text()).get("model_type", "")
        except Exception:
            mt = ""
        if mt.endswith("_text"):
            print(f"[merge]   detected text-only model_type='{mt}' in saved "
                  "config; restoring multimodal config from base",
                  file=sys.stderr)
            shutil.copy2(base_config, out_config)

    n_files = sum(1 for _ in args.output.iterdir())
    print(f"[merge] done — {n_files} files in {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
