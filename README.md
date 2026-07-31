# DriftSentry

Reproducible utilities for **DriftSentry-4B-v1**, a 4B behavioral-safety classifier for AI-agent responses. The repository covers synthetic data generation, Qwen3.5 LoRA training, deterministic benchmark finalization, evaluation, LoRA merging, and H200/vLLM latency measurement.

Public artifacts:

- Model: https://huggingface.co/Tynapse/drift-sentry-4b-v1
- Evaluation benchmark: https://huggingface.co/datasets/Tynapse/drift-sentry-bench-50k-v1
- Code: https://github.com/Tynapse/drift-sentry

## Taxonomy

DriftSentry classifies six agent failure tiers—`goal`, `reasoning`, `environment`, `integration`, `memory`, and `reward`—plus `normal`. The complete definitions and structured output contract are in `taxonomy.yaml`.

The immutable v1 wire-format identifiers (`judge6.label.v1`, related manifest schema IDs, and existing `judge6-sem-*` row IDs) are retained for compatibility and reproducibility. They are legacy machine identifiers, not the public model name.

## Install

Python 3.12 or 3.13 and `uv` are recommended.

```bash
uv sync --extra train --extra eval
```

No credential is embedded in the repository. Generation and OpenAI-compatible evaluation read service credentials from environment variables.

## Generate semantic data

The generator accepts one or more OpenAI-compatible endpoints. `--strict-target-tier` rejects missing or mismatched teacher labels instead of overwriting them.

```bash
export VLLM_API_KEY=replace-with-local-serving-key

uv run python scripts/generate_semantic_synthetic_6tier.py \
  --taxonomy taxonomy.yaml \
  --output-dir artifacts/semantic \
  --num-samples 1000 \
  --base-url http://127.0.0.1:8000/v1 \
  --model teacher-model \
  --concurrency 32 \
  --max-tokens 1536 \
  --case-retries 2 \
  --strict-target-tier
```

## Train a LoRA judge

`configs/drift_sentry_4b_v1_lora.yaml` records the released rank, alpha, dropout, target modules, base revision, and optimization defaults. Point `train_file` and `val_file` at conversation JSONL files whose last assistant message is the gold DriftSentry JSON.

Each JSONL row must contain an `id` and an OpenAI-style `messages` list. The final message must have role `assistant` and contain the gold structured label; only that final assistant span contributes to the SFT loss. The public loader intentionally contains no product-specific or customer-specific prompt adapters.

Single GPU:

```bash
uv run python scripts/sft_train.py \
  --config configs/drift_sentry_4b_v1_lora.yaml
```

Multi-GPU:

```bash
uv run accelerate launch --num_processes 4 scripts/sft_train.py \
  --config configs/drift_sentry_4b_v1_lora.yaml
```

## Evaluate with Transformers

```bash
uv run python scripts/evaluate_drift_sentry_model.py \
  --model Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --adapter /path/to/adapter \
  --data-file /path/to/test.jsonl \
  --output-dir artifacts/eval \
  --run-name drift-sentry-4b-v1 \
  --batch-size 32 \
  --max-new-tokens 512
```

The evaluator reports accuracy, 7-way and six-risk-tier Macro-F1, per-label precision/recall/F1, memory and reward recall, combined high-risk recall, confusion matrix, and parse failures.

## Merge and serve

```bash
uv run python scripts/merge_lora.py \
  --base "$(hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)" \
  --adapter /path/to/adapter \
  --output artifacts/merged \
  --dtype bfloat16

vllm serve artifacts/merged \
  --served-model-name drift-sentry-4b-v1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gdn-prefill-backend triton
```

## Evaluate an OpenAI-compatible endpoint

```bash
export VLLM_API_KEY=replace-with-local-serving-key

uv run python scripts/evaluate_drift_sentry_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model drift-sentry-4b-v1 \
  --data-file /path/to/test.jsonl \
  --predictions artifacts/predictions.jsonl \
  --metrics artifacts/metrics.json \
  --concurrency 128 \
  --max-tokens 512
```

Repeat `--base-url` to distribute requests deterministically across multiple replicas. Use `--resume` only with a preserved predictions JSONL; the last successful record for an ID wins.

## Finalize a 50,000-row benchmark

The finalizer is fail-closed: it enforces class quotas, exact-prompt separation, token-SimHash separation, strict teacher/target agreement, and configured PII/secret patterns.

```bash
uv run python scripts/finalize_drift_sentry_benchmark.py \
  --candidate artifacts/generated/test.jsonl \
  --reference /path/to/train.jsonl \
  --reference /path/to/val.jsonl \
  --reference /path/to/test.jsonl \
  --output artifacts/public-benchmark/test.jsonl \
  --manifest artifacts/public-benchmark/manifest.json
```

For an already selected benchmark, create a deterministic public-safe copy and
an audit report without recording raw matched values:

```bash
uv run python scripts/sanitize_drift_sentry_benchmark.py \
  --input artifacts/public-benchmark/test.jsonl \
  --output artifacts/public-benchmark-safe/test.jsonl \
  --audit artifacts/public-benchmark-safe/content_safety_audit.json \
  --input-label private-pre-sanitization/test.jsonl \
  --output-label data/test.jsonl \
  --teacher-model Qwen/Qwen3.6-27B \
  --teacher-revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --benchmark-name drift-sentry-bench-50k-v1
```

## Audit label consistency

The audit compares each independently assigned target tier with the unmodified
teacher tier retained inside `metadata.raw_teacher_text`. Rows without a valid
raw teacher tier are reported and excluded rather than imputed.

```bash
uv run python scripts/audit_drift_sentry_label_consistency.py \
  --input /path/to/train.jsonl \
  --input /path/to/val.jsonl \
  --input /path/to/test.jsonl \
  --output artifacts/label_consistency_audit.json
```

## Measure operational latency

```bash
export VLLM_API_KEY=replace-with-local-serving-key

uv run python scripts/benchmark_openai_latency.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model drift-sentry-4b-v1 \
  --tokenizer artifacts/merged \
  --sample-jsonl /path/to/test.jsonl \
  --output artifacts/latency.json \
  --hardware-label "NVIDIA H200" \
  --requests 1000 \
  --warmups 20 \
  --target-prompt-tokens 2048 \
  --max-tokens 512
```

This measures natural termination under a 512-token cap. Add `--force-exact-output-tokens` only for a separate full-512-token stress test.

## Released reference result

On the 50,000-row pre-publication source benchmark, the released checkpoint measured 6-tier Macro-F1 `0.8600247542914515`, memory recall `0.9385412291754165`, and reward recall `0.9134817303653927`. The public transform changed 123 prompts only to replace credential-shaped synthetic strings; those rows were not re-inferred. The model repository preserves both input hashes and the exact scope of the reported metrics.

## Limitations

- Training and benchmark labels are synthetic teacher labels rather than fully human-adjudicated gold labels.
- The benchmark does not establish open-world robustness.
- Judge outputs must not be the sole basis for high-impact decisions.
- Safety data can contain harmful or privacy-sensitive synthetic scenarios.

## License and acknowledgement

Code is licensed under Apache-2.0. The public benchmark is separately licensed under CC BY 4.0.

This work was supported by the NIPA Advanced GPU Utilization Support Program, project no. `04-26-03-0029`.
