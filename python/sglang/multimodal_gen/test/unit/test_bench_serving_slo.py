import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _mark_slo_achieved,
    _populate_fixed_slo_ms,
    _populate_slo_ms,
    calculate_metrics,
)
from sglang.multimodal_gen.benchmarks.datasets import (
    RequestFuncInput,
    RequestFuncOutput,
)


def _load_diffusion_dev_module():
    class FakeImage:
        def env(self, *args, **kwargs):
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

    class FakeVolume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

        def commit(self):
            pass

    class FakeSecret:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(fn):
                fn.remote = fn
                return fn

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=SimpleNamespace(from_registry=lambda *args, **kwargs: FakeImage()),
        Secret=FakeSecret,
        Volume=FakeVolume,
        is_local=lambda: False,
    )

    previous_modal = sys.modules.get("modal")
    sys.modules["modal"] = fake_modal
    try:
        repo_root = Path(__file__).resolve().parents[5]
        module_path = repo_root / "scripts" / "modal" / "diffusion_dev.py"
        spec = importlib.util.spec_from_file_location(
            "_test_diffusion_dev", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal


class TestBenchServingSLO(unittest.TestCase):
    def test_fixed_slo_populates_missing_requests_only(self):
        requests = [
            RequestFuncInput(prompt="generated deadline"),
            RequestFuncInput(prompt="trace deadline", slo_ms=1234.0),
        ]

        populated = _populate_fixed_slo_ms(requests, 3000.0)

        self.assertIsNone(requests[0].slo_ms)
        self.assertEqual(populated[0].slo_ms, 3000.0)
        self.assertEqual(populated[1].slo_ms, 1234.0)
        self.assertIs(populated[1], requests[1])

    def test_fixed_slo_marks_pass_and_fail(self):
        request = RequestFuncInput(prompt="deadline", slo_ms=3000.0)
        fast_output = RequestFuncOutput(success=True, latency=2.9, output_count=1)
        slow_output = RequestFuncOutput(success=True, latency=3.1, output_count=1)

        _mark_slo_achieved(request, fast_output)
        _mark_slo_achieved(request, slow_output)

        self.assertTrue(fast_output.slo_achieved)
        self.assertFalse(slow_output.slo_achieved)

        metrics = calculate_metrics(
            [fast_output, slow_output],
            total_duration=6.0,
            requests_list=[request, request],
            args=SimpleNamespace(
                num_outputs_per_prompt=1,
                fixed_slo_ms=3000.0,
                slo_scale=3.0,
            ),
            slo_enabled=True,
        )

        self.assertEqual(metrics["slo_mode"], "fixed")
        self.assertEqual(metrics["fixed_slo_ms"], 3000.0)
        self.assertEqual(metrics["slo_met_success"], 1)
        self.assertEqual(metrics["slo_attainment_rate"], 0.5)

    def test_warmup_scaled_slo_still_assigns_missing_deadline(self):
        request = RequestFuncInput(
            prompt="bench",
            width=16,
            height=16,
            num_inference_steps=2,
        )
        warmup_request = RequestFuncInput(
            prompt="warmup",
            width=16,
            height=16,
            num_inference_steps=1,
        )
        warmup_output = RequestFuncOutput(success=True, latency=0.5)
        args = SimpleNamespace(
            width=16,
            height=16,
            num_frames=None,
            num_inference_steps=2,
            slo_scale=3.0,
            fixed_slo_ms=None,
        )

        populated = _populate_slo_ms([request], [(warmup_request, warmup_output)], args)

        self.assertEqual(populated[0].slo_ms, 3000.0)


class TestModalSLOPresets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diffusion_dev = _load_diffusion_dev_module()

    def test_slo_preset_resolution_and_explicit_override(self):
        preset = self.diffusion_dev._resolve_slo_config("diffserve-t2i-512-5s", 0.0)
        override = self.diffusion_dev._resolve_slo_config(
            "diffserve-t2i-512-5s", 1234.0
        )
        custom = self.diffusion_dev._resolve_slo_config("none", 4321.0)

        self.assertEqual(preset["mode"], "fixed")
        self.assertEqual(preset["fixed_slo_ms"], 5000.0)
        self.assertEqual(preset["preset"], "diffserve-t2i-512-5s")
        self.assertEqual(override["fixed_slo_ms"], 1234.0)
        self.assertEqual(override["source"], "CLI --fixed-slo-ms override")
        self.assertEqual(custom["preset"], "custom-fixed-slo")
        self.assertEqual(custom["fixed_slo_ms"], 4321.0)

        with self.assertRaisesRegex(ValueError, "requires --slo-preset"):
            self.diffusion_dev._resolve_slo_config("none", 0.0)

    def test_dry_run_bench_command_includes_fixed_slo_ms(self):
        result = self.diffusion_dev._saturation_dry_run(
            workload="zimage-turbo-t2i-prod",
            model_path="Efficient-Large-Model/Sana_600M_512px_diffusers",
            height=512,
            width=512,
            num_inference_steps=20,
            num_prompts=16,
            concurrency="1,2",
            slo_preset="genserve-image-3s",
            serve_extra_args="--batching-max-size 32 --warmup-resolutions 512x512",
        )

        first_command = result["bench_cmds"][0]
        self.assertEqual(result["slo"]["fixed_slo_ms"], 3000.0)
        self.assertIn("--fixed-slo-ms", first_command)
        self.assertIn("3000", first_command)
        self.assertIn("--request-rate", first_command)
        self.assertIn("inf", first_command)


if __name__ == "__main__":
    unittest.main()
