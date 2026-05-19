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
python3 -m py_compile scripts/modal/diffusion_dev.py
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench --help
.venv/bin/modal run scripts/modal/diffusion_dev.py::profile_generate --help
```

## What's Enabled By Default

Every `production_bench` run uses a single max-efficiency preset on the H100
80GB target:

- `--enable-torch-compile` and `--warmup --warmup-resolutions <workload res>`
  so the first request does not pay compile cost.
- `--attention-backend fa` (FlashAttention; lossless).
- `--batching-mode dynamic` with per-workload `--batching-max-size` and
  `--batching-delay-ms` (Z-Image-Turbo `8/5ms`, Qwen-Image `4/10ms`,
  FLUX.1-dev `2/10ms`).
- VBench prompts, SLO scoring, warmup requests, and a Poisson arrival sweep at
  concurrency `1,2,4,8`.
- Cache-DiT env knobs (`SGLANG_CACHE_DIT_ENABLED=true` plus conservative
  `FN/BN/WARMUP/RDT/MC` defaults). This is the one lossy optimization in the
  preset; pass `--enable-cache-dit false` for a fully lossless run.

DiT and VAE offloads stay off because the target GPU has the VRAM headroom and
copy overlap is unreliable on T2I image models.

## Dry Run The Production Commands

Prints the exact `sglang serve` and `bench_serving` commands the preset will
launch, without running anything remote.

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --dry-run \
  --source local \
  --workload qwen-image-t2i-prod \
  --num-prompts 20
```

## Realistic Text-To-Image Workloads

### Qwen-Image

```bash
export RUN_ID="qwen-image-t2i-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label qwen-image \
  --workload qwen-image-t2i-prod \
  --num-prompts 80
```

### FLUX.1-dev

```bash
export RUN_ID="flux1-dev-t2i-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label flux1-dev \
  --workload flux1-dev-t2i-prod \
  --num-prompts 40
```

### Z-Image-Turbo

```bash
export RUN_ID="zimage-turbo-t2i-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label zimage-turbo \
  --workload zimage-turbo-t2i-prod \
  --num-prompts 120
```

## Baseline Vs Candidate

Use the same run ID for the baked Docker image baseline and the local checkout
candidate. Artifacts land together so a side-by-side diff is trivial.

```bash
export RUN_ID="qwen-image-compare-$(date -u +%Y%m%d-%H%M%S)"
```

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source image \
  --run-id "$RUN_ID" \
  --label baseline \
  --workload qwen-image-t2i-prod \
  --num-prompts 80
```

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label candidate \
  --workload qwen-image-t2i-prod \
  --num-prompts 80
```

## Cache-DiT Off (Lossless Baseline)

Cache-DiT is on by default for throughput. Disable it when comparing against a
known-good lossless baseline or measuring quality.

```bash
export RUN_ID="qwen-image-no-cache-dit-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label no-cache-dit \
  --workload qwen-image-t2i-prod \
  --enable-cache-dit false \
  --num-prompts 80
```

Tune Cache-DiT aggressively (lossier, faster) by setting the env knobs through
`--serve-extra-args` is not currently supported; override the workload's
`cache_dit_env` in `scripts/modal/diffusion_dev.py` if you need different
thresholds.

## Overrides

The preset is opinionated. Reach for `--serve-extra-args` and
`--bench-extra-args` when you need to override it for one run.

```bash
# Burst traffic instead of Poisson.
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local --workload qwen-image-t2i-prod --num-prompts 80 \
  --traffic burst
```

```bash
# Wider concurrency sweep.
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local --workload qwen-image-t2i-prod --num-prompts 160 \
  --concurrency 1,2,4,8,16
```

```bash
# Per-tier scheduler batch metrics.
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local --workload qwen-image-t2i-prod --num-prompts 80 \
  --serve-extra-args "--enable-batching-metrics"
```

```bash
# Resolution / steps / model override.
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local --workload qwen-image-t2i-prod --num-prompts 80 \
  --model-path Qwen/Qwen-Image --height 768 --width 768 \
  --num-inference-steps 30
```

## Profiling Around A Benchmark

Profiling is intentionally separate from production timing runs. Use it after
the benchmark identifies a workload worth inspecting.

### PyTorch Profiler, Denoising Stage

```bash
export RUN_ID="qwen-profile-denoise-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::profile_generate \
  --source local \
  --run-id "$RUN_ID" \
  --label denoise \
  --model-path Qwen/Qwen-Image \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --num-profiled-timesteps 5
```

### PyTorch Profiler, Full Pipeline

```bash
export RUN_ID="qwen-profile-full-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::profile_generate \
  --source local \
  --run-id "$RUN_ID" \
  --label full-pipeline \
  --model-path Qwen/Qwen-Image \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --num-profiled-timesteps 5 \
  --profile-all-stages
```

### Nsight Systems

```bash
export RUN_ID="qwen-profile-nsys-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::profile_generate \
  --source local \
  --run-id "$RUN_ID" \
  --label nsys \
  --model-path Qwen/Qwen-Image \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --nsys
```

## Download And Inspect Artifacts

```bash
mkdir -p modal-runs
.venv/bin/modal volume get sglang-diffusion-runs "$RUN_ID" "modal-runs/$RUN_ID"
```

```bash
find "modal-runs/$RUN_ID" -maxdepth 2 -type f | sort
```

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
