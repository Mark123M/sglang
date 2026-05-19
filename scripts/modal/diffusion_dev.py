#!/usr/bin/env python3
"""Minimal Modal runner for SGLang Diffusion benchmark iteration.

Examples:

  .venv/bin/modal run scripts/modal/diffusion_dev.py::generate --source image --label baseline
  .venv/bin/modal run scripts/modal/diffusion_dev.py::generate --source local --label candidate
  .venv/bin/modal run scripts/modal/diffusion_dev.py::bench_serving --source local --num-prompts 1 --max-concurrency 1
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

IMAGE_REPO = Path("/sgl-workspace/sglang")
LOCAL_REPO = Path("/workspace/sglang-local")
RUNS_ROOT = Path("/runs")
HF_CACHE_PATH = Path("/cache/huggingface")
SGLANG_CACHE_PATH = Path("/cache/sglang")
SGLANG_DIFFUSION_CACHE_PATH = SGLANG_CACHE_PATH / "sgl_diffusion"


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


@app.function(**REMOTE_OPTIONS)
def _generate_remote(
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
    perf_path = run_dir / f"{label}.json"

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
        _run(cmd, cwd=cwd, env=env, log_path=run_dir / "logs" / f"{label}.generate.log")
        return {
            "run_id": run_dir.name,
            "source": source,
            "label": label,
            "perf_path": str(perf_path),
            "output_dir": str(output_dir),
            "log_path": str(run_dir / "logs" / f"{label}.generate.log"),
        }
    finally:
        _commit_volumes()


@app.function(**REMOTE_OPTIONS)
def _bench_serving_remote(
    source: str,
    label: str,
    run_id: str,
    model_path: str,
    num_prompts: int,
    max_concurrency: int,
    port: int,
    height: int,
    width: int,
    num_inference_steps: int,
    server_timeout_seconds: int,
    serve_extra_args: str,
    bench_extra_args: str,
) -> dict[str, str]:
    cwd, env = _source_env(source)
    run_dir = _run_dir(run_id)
    label = _slug(label)

    server_log_path = run_dir / "logs" / f"{label}.server.log"
    output_path = run_dir / f"{label}.serving.jsonl"
    server_cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if serve_extra_args:
        server_cmd.extend(shlex.split(serve_extra_args))

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

    try:
        _wait_for_health(
            f"http://127.0.0.1:{port}/health",
            process=server_process,
            timeout_seconds=server_timeout_seconds,
            server_log_path=server_log_path,
        )

        bench_cmd = [
            "python",
            "-m",
            "sglang.multimodal_gen.benchmarks.bench_serving",
            "--model",
            model_path,
            "--task",
            "text-to-image",
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(max_concurrency),
            "--port",
            str(port),
            "--output-file",
            str(output_path),
        ]
        if height > 0:
            bench_cmd.extend(["--height", str(height)])
        if width > 0:
            bench_cmd.extend(["--width", str(width)])
        if num_inference_steps > 0:
            bench_cmd.extend(["--num-inference-steps", str(num_inference_steps)])
        if bench_extra_args:
            bench_cmd.extend(shlex.split(bench_extra_args))

        _run(
            bench_cmd,
            cwd=cwd,
            env=env,
            log_path=run_dir / "logs" / f"{label}.bench_serving.log",
        )
        return {
            "run_id": run_dir.name,
            "source": source,
            "label": label,
            "output_path": str(output_path),
            "server_log_path": str(server_log_path),
            "bench_log_path": str(run_dir / "logs" / f"{label}.bench_serving.log"),
        }
    finally:
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=30)
        server_log.close()
        _commit_volumes()


@app.local_entrypoint()
def generate(
    source: str = "local",
    label: str = "candidate",
    run_id: str = "",
    model_path: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    seed: int = 0,
    extra_args: str = "",
) -> None:
    """Run one-off `sglang generate` and write perf/output artifacts to /runs."""
    result = _generate_remote.remote(
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


@app.local_entrypoint()
def bench_serving(
    source: str = "local",
    label: str = "serving",
    run_id: str = "",
    model_path: str = DEFAULT_MODEL,
    num_prompts: int = 1,
    max_concurrency: int = 1,
    port: int = DEFAULT_PORT,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    server_timeout_seconds: int = 1800,
    serve_extra_args: str = "",
    bench_extra_args: str = "",
) -> None:
    """Run `sglang serve` plus diffusion `bench_serving` on one Modal GPU."""
    result = _bench_serving_remote.remote(
        source,
        label,
        run_id or _new_run_id(label, source),
        model_path,
        num_prompts,
        max_concurrency,
        port,
        height,
        width,
        num_inference_steps,
        server_timeout_seconds,
        serve_extra_args,
        bench_extra_args,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
