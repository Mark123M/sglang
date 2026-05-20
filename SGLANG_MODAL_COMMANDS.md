# SGLang Diffusion Modal Commands

Run all commands from the repo root.

## One-Time Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U modal
```

```bash
.venv/bin/modal setup
.venv/bin/modal token info
```

```bash
export HF_TOKEN=hf_...
.venv/bin/modal secret create huggingface HF_TOKEN="$HF_TOKEN"
```

```bash
.venv/bin/modal volume create sglang-hf-cache
.venv/bin/modal volume create sglang-cache
.venv/bin/modal volume create sglang-diffusion-runs
```

## Local Sanity Checks

```bash
python3 -m py_compile scripts/modal/diffusion_dev.py \
  python/sglang/multimodal_gen/benchmarks/bench_serving.py
.venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench --help
.venv/bin/modal run scripts/modal/diffusion_dev.py::generate_perf --help

# Requires a Python env with SGLang benchmark dependencies installed.
PYTHONPATH=python python3 -m sglang.multimodal_gen.benchmarks.bench_serving --help
```

## Silares-Style Saturation Bench

Closed-loop fixed-concurrency saturation: `bench_serving` receives a full
backlog (`request-rate=inf`) while `--max-concurrency=N` caps live in-flight
requests. This mirrors the Silares-style sustained live-concurrency stress
test.

By default, `saturation_bench` uses the paper-derived
`diffserve-sdxl-1024-15s` preset: every request without trace-provided `slo_ms`
gets a fixed `15000 ms` deadline. The summary writes
`max_concurrent_under_slo`, the largest tier where
`slo_attainment_rate >= slo-threshold` (default `0.95`).

Pass `--fixed-slo-ms <deadline>` to override any preset with a concrete custom
deadline.

Each run starts `sglang serve` with torch compile, warmup, dynamic batching, and
the workload's FlashAttention/batching defaults. Cache-DiT is enabled by
default for throughput; pass `--no-enable-cache-dit` for lossless measurement.

```bash
export RUN_ID="qwen-saturation-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label qwen-saturation \
  --workload qwen-image-t2i-prod \
  --num-prompts 128 \
  --concurrency 1,2,4,8,16,32,64 \
  --slo-preset diffserve-sdxl-1024-15s \
  --slo-threshold 0.95
```

The sweep stops early at the first tier whose attainment dips below the
threshold; pass `--no-stop-on-slo-breach` to run every tier. The server-side
scheduler is still capped at the workload's `--batching-max-size`, so to find
the true model ceiling raise it:
`--serve-extra-args "--batching-max-size 16"`.

Dry-run the sweep if you want to confirm the generated `bench_serving` command
includes `--fixed-slo-ms` before spending GPU time:

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench \
  --dry-run \
  --source local \
  --workload qwen-image-t2i-prod \
  --concurrency 1,2 \
  --slo-preset diffserve-sdxl-1024-15s
```

> Modal parses booleans as click-style flags: use `--enable-cache-dit` /
> `--no-enable-cache-dit` and `--stop-on-slo-breach` / `--no-stop-on-slo-breach`.
> Passing `--flag false` errors with "unexpected extra argument (false)".

### Quick Sana 600M Silares-Style Sweeps

Use the smallest supported T2I model when you want fast fixed-SLO saturation
feedback. These commands borrow the `zimage-turbo-t2i-prod` workload template
(small-model batching defaults + Cache-DiT env) and override the model,
resolution, and step count via CLI.

Fast strict 3s image SLO:

```bash
export RUN_ID="sana-600m-silares-3s-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label sana-600m-512-3s \
  --workload zimage-turbo-t2i-prod \
  --model-path Efficient-Large-Model/Sana_600M_512px_diffusers \
  --height 512 \
  --width 512 \
  --num-inference-steps 20 \
  --slo-preset genserve-image-3s \
  --concurrency 1,2,4,8,16,32,64,128 \
  --num-prompts 256 \
  --no-stop-on-slo-breach \
  --serve-extra-args "--batching-max-size 32 --warmup-resolutions 512x512"
```

DiffServe-style 512px 5s SLO:

```bash
export RUN_ID="sana-600m-silares-5s-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label sana-600m-512-5s \
  --workload zimage-turbo-t2i-prod \
  --model-path Efficient-Large-Model/Sana_600M_512px_diffusers \
  --height 512 \
  --width 512 \
  --num-inference-steps 20 \
  --slo-preset diffserve-t2i-512-5s \
  --concurrency 1,2,4,8,16,32,64,128 \
  --num-prompts 256 \
  --no-stop-on-slo-breach \
  --serve-extra-args "--batching-max-size 32 --warmup-resolutions 512x512"
```

For a tiny first smoke pass, keep the same command and change only:
`--concurrency 1,2 --num-prompts 16`.

The `--warmup-resolutions 512x512` override is still important: the
`zimage-turbo-t2i-prod` template warms up at `1024x1024`, but these runs use
`512x512`. With fixed SLOs, this avoids charging the first 512x512 compile to
serving latency rather than affecting the SLO deadline itself.

## SGLang Performance Contribution Commands

The SGLang diffusion contributing guide asks for a single-generation
performance dump before and after a performance-sensitive change:

```bash
sglang generate \
  --model-path <model> \
  --prompt "A benchmark prompt" \
  --perf-dump-path baseline.json

sglang generate \
  --model-path <model> \
  --prompt "A benchmark prompt" \
  --perf-dump-path new.json

python python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  baseline.json new.json
```

Use the Modal helper to run the same flow against the baked image baseline and
your local checkout candidate:

```bash
export RUN_ID="qwen-generate-perf-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::generate_perf \
  --source image \
  --run-id "$RUN_ID" \
  --label baseline \
  --model-path Qwen/Qwen-Image \
  --prompt "A benchmark prompt" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20

.venv/bin/modal run scripts/modal/diffusion_dev.py::generate_perf \
  --source local \
  --run-id "$RUN_ID" \
  --label new \
  --model-path Qwen/Qwen-Image \
  --prompt "A benchmark prompt" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20
```

## Download And Inspect Artifacts

```bash
mkdir -p modal-runs
.venv/bin/modal volume get sglang-diffusion-runs "$RUN_ID" "modal-runs/$RUN_ID"
```

```bash
find "modal-runs/$RUN_ID" -maxdepth 2 -type f | sort
```

For saturation sweeps:

```bash
python3 -m json.tool "modal-runs/$RUN_ID/summary.json"
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

run_dir = Path("modal-runs") / Path(__import__("os").environ["RUN_ID"])
summary = json.loads((run_dir / "summary.json").read_text())
for row in summary["benchmarks"]:
    print(
        row["concurrency"],
        row["request_rate"],
        row["throughput_qps"],
        row["latency_p50_s"],
        row["latency_p95_s"],
        row["latency_p99_s"],
        row["slo_attainment_rate"],
    )
PY
```

Benchmark JSON files are written under the run directory as one file per
concurrency tier, for example `qwen-image.c1.json`, `qwen-image.c2.json`,
`qwen-image.c4.json`, and `qwen-image.c8.json`.

For `generate_perf` runs:

```bash
python python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  "modal-runs/$RUN_ID/baseline.perf.json" \
  "modal-runs/$RUN_ID/new.perf.json"
```
