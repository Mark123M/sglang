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
.venv/bin/modal run scripts/modal/diffusion_dev.py::generate --help
.venv/bin/modal run scripts/modal/diffusion_dev.py::bench_serving --help
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench --help
.venv/bin/modal run scripts/modal/diffusion_dev.py::profile_generate --help
```

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::generate \
  --source local \
  --label smoke \
  --height 512 \
  --width 512 \
  --num-inference-steps 2
```

## Dry Run The Production Commands

Use this first to print the exact `sglang serve` and diffusion `bench_serving`
commands without launching the remote benchmark.

```bash
.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --dry-run \
  --source local \
  --workload qwen-image-t2i-prod \
  --num-prompts 20
```

## Realistic Text-To-Image Workloads

The production benchmark uses SGLang's diffusion serving benchmark with VBench
prompts, warmup requests, SLO scoring, and a concurrency sweep. By default it
uses Poisson arrivals and concurrency `1,2,4,8`.

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
candidate. This keeps artifacts together while isolating source changes.

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

## Traffic Shapes

### Poisson Arrivals

This is the default production-style traffic shape.

```bash
export RUN_ID="qwen-poisson-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label poisson \
  --workload qwen-image-t2i-prod \
  --traffic poisson \
  --request-rate auto \
  --num-prompts 80
```

### Burst Arrivals

Use this to stress scheduler behavior by sending each concurrency tier as fast
as the benchmark can issue requests.

```bash
export RUN_ID="qwen-burst-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label burst \
  --workload qwen-image-t2i-prod \
  --traffic burst \
  --num-prompts 80
```

### Fixed Request Rate

```bash
export RUN_ID="qwen-fixed-rate-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label fixed-rate \
  --workload qwen-image-t2i-prod \
  --traffic poisson \
  --request-rate 0.2 \
  --num-prompts 80
```

## Concurrency Sweeps

```bash
export RUN_ID="qwen-concurrency-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label c1-c16 \
  --workload qwen-image-t2i-prod \
  --concurrency 1,2,4,8,16 \
  --num-prompts 160
```

## Dynamic Batching Experiments

The default production benchmark leaves scheduler metrics and extra logging off
to avoid perturbing the run. Enable dynamic batching explicitly when testing it.

```bash
export RUN_ID="qwen-dynamic-batching-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label dynbatch \
  --workload qwen-image-t2i-prod \
  --concurrency 1,2,4,8,16 \
  --num-prompts 160 \
  --enable-dynamic-batching \
  --batching-max-size 8 \
  --batching-delay-ms 5
```

Add scheduler batch logs only when you want those logs more than a clean timing
run:

```bash
export RUN_ID="qwen-dynamic-batching-metrics-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label dynbatch-metrics \
  --workload qwen-image-t2i-prod \
  --concurrency 1,2,4,8,16 \
  --num-prompts 160 \
  --enable-dynamic-batching \
  --batching-max-size 8 \
  --batching-delay-ms 5 \
  --serve-extra-args "--enable-batching-metrics"
```

## Override Resolution, Steps, Or Model

```bash
export RUN_ID="qwen-768-steps-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label qwen-768-30steps \
  --workload qwen-image-t2i-prod \
  --height 768 \
  --width 768 \
  --num-inference-steps 30 \
  --num-prompts 80
```

```bash
export RUN_ID="custom-t2i-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::production_bench \
  --source local \
  --run-id "$RUN_ID" \
  --label custom-model \
  --workload qwen-image-t2i-prod \
  --model-path Qwen/Qwen-Image \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --num-prompts 80
```

## Low-Level Serving Smoke Test

Use this only when you want a single serving benchmark run instead of the
production sweep.

```bash
export RUN_ID="qwen-serving-smoke-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/modal run scripts/modal/diffusion_dev.py::bench_serving \
  --source local \
  --run-id "$RUN_ID" \
  --label serving \
  --num-prompts 4 \
  --max-concurrency 1 \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --bench-extra-args "--dataset vbench --slo --disable-tqdm"
```

## Profiling Around A Benchmark

Use profiling after the production benchmark identifies a workload worth
inspecting. Profiling is intentionally separate from production timing runs.

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
