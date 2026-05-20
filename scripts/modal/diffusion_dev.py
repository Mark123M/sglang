#!/usr/bin/env python3
"""Minimal Modal runner for SGLang Diffusion benchmark iteration.

Examples:

  .venv/bin/modal run scripts/modal/diffusion_dev.py::saturation_bench \
    --source local --num-prompts 20
  .venv/bin/modal run scripts/modal/diffusion_dev.py::generate_perf \
    --source local --label new
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "sglang-diffusion-dev"
DEFAULT_GPU = "H100!:1"
DEFAULT_MODEL = "Qwen/Qwen-Image"
DEFAULT_PROMPT = "A logo With Bold Large text: SGL Diffusion"
DEFAULT_PORT = 30010
REMOTE_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_SATURATION_CONCURRENCY = "1,2,4,8,16,32,64"
DEFAULT_SATURATION_SLO_PRESET = "diffserve-sdxl-1024-15s"

IMAGE_REPO = Path("/sgl-workspace/sglang")
LOCAL_REPO = Path("/workspace/sglang-local")
RUNS_ROOT = Path("/runs")
HF_CACHE_PATH = Path("/cache/huggingface")
SGLANG_CACHE_PATH = Path("/cache/sglang")
SGLANG_DIFFUSION_CACHE_PATH = SGLANG_CACHE_PATH / "sgl_diffusion"

DEFAULT_CACHE_DIT_ENV = {
    "SGLANG_CACHE_DIT_ENABLED": "true",
    "SGLANG_CACHE_DIT_FN": "1",
    "SGLANG_CACHE_DIT_BN": "0",
    "SGLANG_CACHE_DIT_WARMUP": "4",
    "SGLANG_CACHE_DIT_RDT": "0.24",
    "SGLANG_CACHE_DIT_MC": "3",
}

T2I_WORKLOADS = {
    "qwen-image-t2i-prod": {
        "model_path": "Qwen/Qwen-Image",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 20,
        "concurrency": DEFAULT_SATURATION_CONCURRENCY,
        "warmup_resolutions": "1024x1024",
        "attention_backend": "fa",
        "batching_max_size": 4,
        "batching_delay_ms": 10.0,
        "cache_dit_env": dict(DEFAULT_CACHE_DIT_ENV),
    },
    "flux1-dev-t2i-prod": {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 28,
        "concurrency": DEFAULT_SATURATION_CONCURRENCY,
        "warmup_resolutions": "1024x1024",
        "attention_backend": "fa",
        "batching_max_size": 2,
        "batching_delay_ms": 10.0,
        "cache_dit_env": dict(DEFAULT_CACHE_DIT_ENV),
    },
    "zimage-turbo-t2i-prod": {
        "model_path": "Tongyi-MAI/Z-Image-Turbo",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 8,
        "concurrency": DEFAULT_SATURATION_CONCURRENCY,
        "warmup_resolutions": "1024x1024",
        "attention_backend": "fa",
        "batching_max_size": 8,
        "batching_delay_ms": 5.0,
        "cache_dit_env": dict(DEFAULT_CACHE_DIT_ENV),
    },
}

SLO_PRESETS = {
    "genserve-image-3s": {
        "name": "GENSERVE image SLO",
        "fixed_slo_ms": 3000.0,
        "source": "GENSERVE image_slo=3.0s for mixed image/video diffusion serving",
        "modality": "image",
    },
    "diffserve-t2i-512-5s": {
        "name": "DiffServe 512px T2I SLO",
        "fixed_slo_ms": 5000.0,
        "source": "DiffServe 512x512 SD-Turbo/SDv1.5 cascades use a 5s SLO",
        "modality": "image",
    },
    "diffserve-sdxl-1024-15s": {
        "name": "DiffServe SDXL 1024px SLO",
        "fixed_slo_ms": 15000.0,
        "source": "DiffServe 1024x1024 SDXL cascade uses a 15s SLO",
        "modality": "image",
    },
    "genserve-video-60s": {
        "name": "GENSERVE video SLO",
        "fixed_slo_ms": 60000.0,
        "source": "GENSERVE video_slo=60.0s for mixed image/video diffusion serving",
        "modality": "video",
    },
}


def _ignore_local_path(path: Path) -> bool:
    ignored_parts = {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "tl_out",
    }
    if any(part in ignored_parts for part in path.parts):
        return True
    return path.suffix in {".egg-info", ".jsonl", ".log", ".o", ".pdf", ".pyc", ".so"}


app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("sglang-hf-cache", create_if_missing=True)
sglang_cache_vol = modal.Volume.from_name("sglang-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("sglang-diffusion-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

image = modal.Image.from_registry("lmsysorg/sglang:dev").env(
    {
        "HF_HOME": str(HF_CACHE_PATH),
        "HUGGINGFACE_HUB_CACHE": str(HF_CACHE_PATH / "hub"),
        "PYTHONUNBUFFERED": "1",
        "SGLANG_CACHE_DIR": str(SGLANG_CACHE_PATH),
        "SGLANG_DIFFUSION_CACHE_ROOT": str(SGLANG_DIFFUSION_CACHE_PATH),
    }
)

if modal.is_local():
    repo_root = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        repo_root,
        str(LOCAL_REPO),
        copy=False,
        ignore=_ignore_local_path,
    )

REMOTE_OPTIONS = dict(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=REMOTE_TIMEOUT_SECONDS,
    secrets=[hf_secret],
    volumes={
        str(HF_CACHE_PATH): hf_cache_vol,
        str(SGLANG_CACHE_PATH): sglang_cache_vol,
        str(RUNS_ROOT): runs_vol,
    },
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _new_run_id(label: str, source: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_slug(source)}-{_slug(label)}-{uuid.uuid4().hex[:8]}"


def _source_env(source: str) -> tuple[Path, dict[str, str]]:
    if source not in {"image", "local"}:
        raise ValueError("source must be 'image' or 'local'")

    cwd = IMAGE_REPO if source == "image" else LOCAL_REPO
    if not cwd.exists():
        raise FileNotFoundError(f"source checkout does not exist: {cwd}")

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE_PATH)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_PATH / "hub")
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_CACHE_DIR"] = str(SGLANG_CACHE_PATH)
    env["SGLANG_DIFFUSION_CACHE_ROOT"] = str(SGLANG_DIFFUSION_CACHE_PATH)

    if source == "local":
        local_python = str(LOCAL_REPO / "python")
        env["PYTHONPATH"] = (
            local_python
            if not env.get("PYTHONPATH")
            else f"{local_python}:{env['PYTHONPATH']}"
        )

    return cwd, env


def _run_dir(run_id: str) -> Path:
    path = RUNS_ROOT / _slug(run_id)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "outputs").mkdir(parents=True, exist_ok=True)
    return path


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(cmd)}")
    print(f"cwd={cwd}")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.write(f"cwd={cwd}\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()
        log_file.write(f"\nreturncode={return_code}\n")

    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}; see {log_path}")


def _wait_for_health(
    url: str,
    *,
    process: subprocess.Popen[str],
    timeout_seconds: int,
    server_log_path: Path,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""

    while time.time() < deadline:
        if process.poll() is not None:
            tail = ""
            if server_log_path.exists():
                tail = server_log_path.read_text(encoding="utf-8")[-4000:]
            raise RuntimeError(
                f"server exited before becoming healthy; see {server_log_path}\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(f"server did not become healthy at {url}: {last_error}")


def _commit_volumes() -> None:
    runs_vol.commit()
    hf_cache_vol.commit()
    sglang_cache_vol.commit()


def _split_csv_ints(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_t2i_workload(
    workload: str,
    *,
    model_path: str,
    height: int,
    width: int,
    num_inference_steps: int,
    concurrency: str,
) -> dict:
    if workload not in T2I_WORKLOADS:
        raise ValueError(
            f"unknown workload {workload!r}; choose one of {sorted(T2I_WORKLOADS)}"
        )
    cfg = dict(T2I_WORKLOADS[workload])
    if "cache_dit_env" in cfg:
        cfg["cache_dit_env"] = dict(cfg["cache_dit_env"])
    if model_path:
        cfg["model_path"] = model_path
    if height > 0:
        cfg["height"] = height
    if width > 0:
        cfg["width"] = width
    if num_inference_steps > 0:
        cfg["num_inference_steps"] = num_inference_steps
    if concurrency:
        cfg["concurrency"] = concurrency
    cfg["workload"] = workload
    cfg["concurrency_values"] = _split_csv_ints(str(cfg["concurrency"]))
    return cfg


def _resolve_slo_config(slo_preset: str, fixed_slo_ms: float) -> dict:
    if fixed_slo_ms < 0:
        raise ValueError(f"fixed_slo_ms must be non-negative, got {fixed_slo_ms}")

    normalized_preset = (slo_preset or "").strip()
    if normalized_preset.lower() in {"", "none"}:
        normalized_preset = ""

    preset_cfg = {}
    if normalized_preset:
        if normalized_preset not in SLO_PRESETS:
            choices = ", ".join(sorted(SLO_PRESETS))
            raise ValueError(
                f"unknown SLO preset {slo_preset!r}; choose one of: {choices}, "
                "or pass --fixed-slo-ms"
            )
        preset_cfg = SLO_PRESETS[normalized_preset]

    if fixed_slo_ms > 0:
        return {
            "mode": "fixed",
            "preset": normalized_preset or "custom-fixed-slo",
            "name": preset_cfg.get("name", "Custom fixed SLO"),
            "source": (
                "CLI --fixed-slo-ms override"
                if normalized_preset
                else "CLI --fixed-slo-ms"
            ),
            "preset_source": preset_cfg.get("source", ""),
            "modality": preset_cfg.get("modality", "image"),
            "fixed_slo_ms": float(fixed_slo_ms),
        }

    if not normalized_preset:
        raise ValueError("saturation_bench requires --slo-preset or --fixed-slo-ms")

    return {
        "mode": "fixed",
        "preset": normalized_preset,
        "name": preset_cfg["name"],
        "source": preset_cfg["source"],
        "preset_source": preset_cfg["source"],
        "modality": preset_cfg["modality"],
        "fixed_slo_ms": float(preset_cfg["fixed_slo_ms"]),
    }


def _fixed_slo_ms_arg(slo_config: dict) -> float | None:
    fixed_slo_ms = slo_config.get("fixed_slo_ms")
    if fixed_slo_ms is None:
        return None
    return float(fixed_slo_ms)


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _apply_cache_dit_env(
    env: dict[str, str], cache_dit_env: dict[str, str] | None
) -> None:
    if not cache_dit_env:
        return
    for key, value in cache_dit_env.items():
        # Caller-set values win so a user-provided env override is honored.
        env.setdefault(key, str(value))


def _build_server_cmd(
    *,
    model_path: str,
    port: int,
    warmup_resolutions: str,
    attention_backend: str,
    batching_max_size: int,
    batching_delay_ms: float,
    serve_extra_args: str,
) -> list[str]:
    extra = shlex.split(serve_extra_args) if serve_extra_args else []
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if not _has_arg(extra, "--warmup"):
        cmd.append("--warmup")
    if warmup_resolutions and not _has_arg(extra, "--warmup-resolutions"):
        cmd.extend(["--warmup-resolutions", warmup_resolutions])
    if not _has_arg(extra, "--enable-torch-compile"):
        cmd.append("--enable-torch-compile")
    if attention_backend and not _has_arg(extra, "--attention-backend"):
        cmd.extend(["--attention-backend", attention_backend])
    if not _has_arg(extra, "--batching-mode"):
        cmd.extend(["--batching-mode", "dynamic"])
    if not _has_arg(extra, "--batching-max-size"):
        cmd.extend(["--batching-max-size", str(batching_max_size)])
    if not _has_arg(extra, "--batching-delay-ms"):
        cmd.extend(["--batching-delay-ms", str(batching_delay_ms)])
    cmd.extend(extra)
    return cmd


def _build_saturation_bench_cmd(
    *,
    model_path: str,
    port: int,
    output_path: Path,
    num_prompts: int,
    max_concurrency: int,
    warmup_requests: int,
    height: int,
    width: int,
    num_inference_steps: int,
    bench_extra_args: str,
    fixed_slo_ms: float | None,
) -> list[str]:
    extra = shlex.split(bench_extra_args) if bench_extra_args else []
    cmd = [
        "python",
        "-m",
        "sglang.multimodal_gen.benchmarks.bench_serving",
        "--model",
        model_path,
        "--task",
        "text-to-image",
        "--dataset",
        "vbench",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        "inf",
        "--warmup-requests",
        str(warmup_requests),
        "--port",
        str(port),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-inference-steps",
        str(num_inference_steps),
        "--output-file",
        str(output_path),
    ]
    if not _has_arg(extra, "--slo"):
        cmd.append("--slo")
    if fixed_slo_ms is not None and fixed_slo_ms > 0 and not _has_arg(
        extra, "--fixed-slo-ms"
    ):
        cmd.extend(["--fixed-slo-ms", f"{fixed_slo_ms:g}"])
    if not _has_arg(extra, "--disable-tqdm"):
        cmd.append("--disable-tqdm")
    cmd.extend(extra)
    return cmd


def _write_saturation_summary(
    *,
    run_dir: Path,
    workload_cfg: dict,
    bench_runs: list[dict],
    slo_threshold: float,
    slo_config: dict,
) -> tuple[str, int]:
    benchmarks = []
    max_concurrent_under_slo = 0
    for run in bench_runs:
        metrics = _load_json(Path(run["output_path"]))
        attainment = float(metrics.get("slo_attainment_rate", 0))
        passed = attainment >= slo_threshold
        if passed:
            max_concurrent_under_slo = max(max_concurrent_under_slo, run["concurrency"])
        benchmarks.append(
            {
                "concurrency": run["concurrency"],
                "request_rate": run["request_rate"],
                "total_requests": run["num_prompts"],
                "completed_requests": metrics.get("completed_requests", 0),
                "failed_requests": metrics.get("failed_requests", 0),
                "throughput_qps": metrics.get("throughput_qps", 0),
                "latency_p50_s": metrics.get(
                    "latency_p50", metrics.get("latency_median", 0)
                ),
                "latency_p95_s": metrics.get("latency_p95", 0),
                "latency_p99_s": metrics.get("latency_p99", 0),
                "slo_attainment_rate": attainment,
                "slo_passed": passed,
                "slo_mode": metrics.get("slo_mode", slo_config["mode"]),
                "fixed_slo_ms": metrics.get("fixed_slo_ms", slo_config["fixed_slo_ms"]),
                "raw_output_path": run["output_path"],
            }
        )
    summary = {
        "workload": workload_cfg,
        "slo": slo_config,
        "slo_threshold": slo_threshold,
        "max_concurrent_under_slo": max_concurrent_under_slo,
        "benchmarks": benchmarks,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(summary_path), max_concurrent_under_slo


def _saturation_dry_run(
    *,
    workload: str = "qwen-image-t2i-prod",
    model_path: str = "",
    height: int = 0,
    width: int = 0,
    num_inference_steps: int = 0,
    num_prompts: int = 128,
    concurrency: str = DEFAULT_SATURATION_CONCURRENCY,
    port: int = DEFAULT_PORT,
    warmup_requests: int = 1,
    serve_extra_args: str = "",
    bench_extra_args: str = "",
    enable_cache_dit: bool = True,
    slo_preset: str = DEFAULT_SATURATION_SLO_PRESET,
    fixed_slo_ms: float = 0.0,
) -> dict:
    cfg = _resolve_t2i_workload(
        workload,
        model_path=model_path,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        concurrency=concurrency,
    )
    slo_config = _resolve_slo_config(slo_preset, fixed_slo_ms)
    server_cmd = _build_server_cmd(
        model_path=cfg["model_path"],
        port=port,
        warmup_resolutions=str(cfg.get("warmup_resolutions", "")),
        attention_backend=str(cfg.get("attention_backend", "")),
        batching_max_size=int(cfg["batching_max_size"]),
        batching_delay_ms=float(cfg["batching_delay_ms"]),
        serve_extra_args=serve_extra_args,
    )
    cache_dit_env = dict(cfg.get("cache_dit_env") or {}) if enable_cache_dit else {}
    bench_cmds = []
    for conc in cfg["concurrency_values"]:
        bench_cmds.append(
            _build_saturation_bench_cmd(
                model_path=cfg["model_path"],
                port=port,
                output_path=Path(f"/runs/dry-run/c{conc}.json"),
                num_prompts=num_prompts,
                max_concurrency=conc,
                warmup_requests=warmup_requests,
                height=cfg["height"],
                width=cfg["width"],
                num_inference_steps=cfg["num_inference_steps"],
                bench_extra_args=bench_extra_args,
                fixed_slo_ms=_fixed_slo_ms_arg(slo_config),
            )
        )
    return {
        "workload": cfg,
        "slo": slo_config,
        "server_cmd": server_cmd,
        "bench_cmds": bench_cmds,
        "cache_dit_env": cache_dit_env,
    }


@app.function(**REMOTE_OPTIONS)
def _saturation_bench_remote(
    source: str,
    label: str,
    run_id: str,
    workload: str,
    model_path: str,
    height: int,
    width: int,
    num_inference_steps: int,
    num_prompts: int,
    concurrency: str,
    port: int,
    warmup_requests: int,
    server_timeout_seconds: int,
    serve_extra_args: str,
    bench_extra_args: str,
    enable_cache_dit: bool,
    slo_threshold: float,
    stop_on_slo_breach: bool,
    slo_preset: str,
    fixed_slo_ms: float,
) -> dict[str, object]:
    cwd, env = _source_env(source)
    run_dir = _run_dir(run_id)
    label = _slug(label)
    cfg = _resolve_t2i_workload(
        workload,
        model_path=model_path,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        concurrency=concurrency,
    )
    slo_config = _resolve_slo_config(slo_preset, fixed_slo_ms)
    if enable_cache_dit:
        _apply_cache_dit_env(env, cfg.get("cache_dit_env"))

    server_log_path = run_dir / "logs" / f"{label}.server.log"
    server_cmd = _build_server_cmd(
        model_path=cfg["model_path"],
        port=port,
        warmup_resolutions=str(cfg.get("warmup_resolutions", "")),
        attention_backend=str(cfg.get("attention_backend", "")),
        batching_max_size=int(cfg["batching_max_size"]),
        batching_delay_ms=float(cfg["batching_delay_ms"]),
        serve_extra_args=serve_extra_args,
    )

    print(f"$ {shlex.join(server_cmd)}")
    print(f"cwd={cwd}")
    server_log = server_log_path.open("w", encoding="utf-8")
    server_log.write(f"$ {shlex.join(server_cmd)}\n")
    server_log.write(f"cwd={cwd}\n")
    server_log.flush()
    server_process = subprocess.Popen(
        server_cmd,
        cwd=str(cwd),
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    bench_runs: list[dict] = []
    try:
        try:
            _wait_for_health(
                f"http://127.0.0.1:{port}/health",
                process=server_process,
                timeout_seconds=server_timeout_seconds,
                server_log_path=server_log_path,
            )

            for conc in cfg["concurrency_values"]:
                output_path = run_dir / f"{label}.c{conc}.json"
                bench_cmd = _build_saturation_bench_cmd(
                    model_path=cfg["model_path"],
                    port=port,
                    output_path=output_path,
                    num_prompts=num_prompts,
                    max_concurrency=conc,
                    warmup_requests=warmup_requests,
                    height=cfg["height"],
                    width=cfg["width"],
                    num_inference_steps=cfg["num_inference_steps"],
                    bench_extra_args=bench_extra_args,
                    fixed_slo_ms=_fixed_slo_ms_arg(slo_config),
                )
                _run(
                    bench_cmd,
                    cwd=cwd,
                    env=env,
                    log_path=run_dir / "logs" / f"{label}.c{conc}.bench.log",
                )
                bench_runs.append(
                    {
                        "concurrency": conc,
                        "request_rate": "inf",
                        "num_prompts": num_prompts,
                        "output_path": str(output_path),
                    }
                )
                metrics = _load_json(output_path)
                attainment = float(metrics.get("slo_attainment_rate", 0))
                if stop_on_slo_breach and attainment < slo_threshold:
                    print(
                        f"SLO breached at c={conc}: attainment={attainment:.3f} "
                        f"< threshold={slo_threshold}. Stopping sweep early."
                    )
                    break
        finally:
            if server_process.poll() is None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait(timeout=30)
            server_log.close()

        summary_path, max_under_slo = _write_saturation_summary(
            run_dir=run_dir,
            workload_cfg=cfg,
            bench_runs=bench_runs,
            slo_threshold=slo_threshold,
            slo_config=slo_config,
        )
        return {
            "run_id": run_dir.name,
            "source": source,
            "label": label,
            "workload": workload,
            "slo": slo_config,
            "slo_threshold": slo_threshold,
            "max_concurrent_under_slo": max_under_slo,
            "bench_runs": bench_runs,
            "artifacts": {
                "summary": summary_path,
                "server_log": str(server_log_path),
            },
        }
    finally:
        _commit_volumes()


@app.function(**REMOTE_OPTIONS)
def _generate_perf_remote(
    source: str,
    label: str,
    run_id: str,
    model_path: str,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    extra_args: str,
) -> dict[str, str]:
    cwd, env = _source_env(source)
    run_dir = _run_dir(run_id)
    label = _slug(label)
    output_dir = run_dir / "outputs" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_path = run_dir / f"{label}.perf.json"

    cmd = [
        "sglang",
        "generate",
        "--model-path",
        model_path,
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-inference-steps",
        str(num_inference_steps),
        "--save-output",
        "--output-path",
        str(output_dir),
        "--output-file-name",
        label,
        "--perf-dump-path",
        str(perf_path),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    try:
        _run(
            cmd,
            cwd=cwd,
            env=env,
            log_path=run_dir / "logs" / f"{label}.generate_perf.log",
        )
        return {
            "run_id": run_dir.name,
            "source": source,
            "label": label,
            "perf_path": str(perf_path),
            "output_dir": str(output_dir),
            "log_path": str(run_dir / "logs" / f"{label}.generate_perf.log"),
        }
    finally:
        _commit_volumes()


@app.local_entrypoint()
def saturation_bench(
    source: str = "local",
    label: str = "saturation",
    run_id: str = "",
    workload: str = "qwen-image-t2i-prod",
    model_path: str = "",
    height: int = 0,
    width: int = 0,
    num_inference_steps: int = 0,
    num_prompts: int = 128,
    concurrency: str = DEFAULT_SATURATION_CONCURRENCY,
    port: int = DEFAULT_PORT,
    warmup_requests: int = 1,
    server_timeout_seconds: int = 1800,
    serve_extra_args: str = "",
    bench_extra_args: str = "",
    enable_cache_dit: bool = True,
    slo_threshold: float = 0.95,
    stop_on_slo_breach: bool = True,
    slo_preset: str = DEFAULT_SATURATION_SLO_PRESET,
    fixed_slo_ms: float = 0.0,
    dry_run: bool = False,
) -> None:
    """Closed-loop saturation sweep: find max concurrency that holds SLO."""
    if dry_run:
        result = _saturation_dry_run(
            workload=workload,
            model_path=model_path,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_prompts=num_prompts,
            concurrency=concurrency,
            port=port,
            warmup_requests=warmup_requests,
            serve_extra_args=serve_extra_args,
            bench_extra_args=bench_extra_args,
            enable_cache_dit=enable_cache_dit,
            slo_preset=slo_preset,
            fixed_slo_ms=fixed_slo_ms,
        )
        result["slo_threshold"] = slo_threshold
        result["stop_on_slo_breach"] = stop_on_slo_breach
    else:
        result = _saturation_bench_remote.remote(
            source,
            label,
            run_id or _new_run_id(label, source),
            workload,
            model_path,
            height,
            width,
            num_inference_steps,
            num_prompts,
            concurrency,
            port,
            warmup_requests,
            server_timeout_seconds,
            serve_extra_args,
            bench_extra_args,
            enable_cache_dit,
            slo_threshold,
            stop_on_slo_breach,
            slo_preset,
            fixed_slo_ms,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def generate_perf(
    source: str = "local",
    label: str = "new",
    run_id: str = "",
    model_path: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    seed: int = 0,
    extra_args: str = "",
) -> None:
    """Run one `sglang generate` perf dump for SGLang performance reports."""
    result = _generate_perf_remote.remote(
        source,
        label,
        run_id or _new_run_id(label, source),
        model_path,
        prompt,
        height,
        width,
        num_inference_steps,
        seed,
        extra_args,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
