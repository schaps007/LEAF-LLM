"""Deterministic regression checks using fake inference and energy backends; no downloads."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnergyTests(unittest.TestCase):
    def test_energy_does_not_depend_on_tracker_carbon(self):
        class Tracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = types.SimpleNamespace(
                    energy_consumed=0.002
                )

            def start(self):
                pass

            def stop(self):
                # Deliberately unrelated to energy and configured intensity.
                return 999

        with patch.dict(
            sys.modules,
            {
                "codecarbon": types.SimpleNamespace(
                    OfflineEmissionsTracker=Tracker
                )
            },
        ):
            module = load(
                "test_energy",
                "metrics/energy.py",
            )
            monitor = module.HybridEnergyMonitor(
                carbon_intensity=0.4
            )
            monitor.start()
            data = monitor.stop()

        self.assertEqual(data["energy_kwh"], 0.002)
        self.assertAlmostEqual(
            data["carbon_kg"],
            0.0008,
        )

    def test_invalid_hardware_energy_is_not_zero_success(self):
        module = load(
            "test_energy",
            "metrics/energy.py",
        )

        monitor = module.HybridEnergyMonitor.__new__(
            module.HybridEnergyMonitor
        )
        monitor.method = "hardware"
        monitor.start_time = 0
        monitor.start_energy = 0
        monitor._get_gpu_energy = lambda: 0

        with self.assertRaises(RuntimeError):
            monitor.stop()


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        return self

    def sum(self):
        return self.data.sum()

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def tolist(self):
        return self.data.tolist()


class InferenceTests(unittest.TestCase):
    def setUp(self):
        from contextlib import nullcontext

        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False
            ),
            inference_mode=nullcontext,
            float32="fp32",
            float16="fp16",
        )

        self.transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(),
            AutoModelForCausalLM=types.SimpleNamespace(),
            BitsAndBytesConfig=lambda **kw: kw,
        )

        with patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "transformers": self.transformers,
            },
        ):
            self.module = load(
                "test_inference",
                "experiments/run_experiment.py",
            )

    def test_eos_and_padding_counts(self):
        self.assertEqual(
            self.module.count_output_tokens(
                [[4, 2, 2], [4, 5, 6]],
                2,
            ),
            5,
        )

        self.assertEqual(
            self.module.count_output_tokens(
                [[4, 3, 0]],
                [2, 3],
            ),
            2,
        )

    def test_unsupported_cpu_quantization_fails(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.module.load_model_with_fallback(
                "fake",
                "int4",
                "cpu",
            )

    def test_gpu_request_does_not_fall_back(self):
        result = self.module.run_single_experiment(
            {"max_new_tokens": 2},
            "fake",
            ["p"],
            1,
            hardware="gpu",
        )

        self.assertEqual(result["status"], "failed")
        self.assertNotIn("energy_per_token", result)

    def test_actual_batches_single_generation_and_phase_hooks(self):
        calls = []

        class Tokenizer:
            pad_token_id = 2
            eos_token_id = 2

            def __call__(self, prompts, **kwargs):
                return {
                    "input_ids": Tensor(
                        [[1, 1]] * len(prompts)
                    ),
                    "attention_mask": Tensor(
                        [[1, 1]] * len(prompts)
                    ),
                }

        class Model:
            pre = None
            post = None

            def eval(self):
                pass

            def parameters(self):
                return iter(
                    [
                        types.SimpleNamespace(
                            device="cpu",
                            numel=lambda: 10,
                        )
                    ]
                )

            def register_forward_pre_hook(self, hook):
                self.pre = hook
                return types.SimpleNamespace(
                    remove=lambda: setattr(self, "pre", None)
                )

            def register_forward_hook(self, hook):
                self.post = hook
                return types.SimpleNamespace(
                    remove=lambda: setattr(self, "post", None)
                )

            def generate(self, input_ids, **kwargs):
                calls.append(input_ids.shape[0])

                if self.pre:
                    self.pre(self, ())
                    self.post(self, (), None)

                return Tensor(
                    [[1, 1, 4, 2]] * input_ids.shape[0]
                )

        class Monitor:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                return {
                    "energy_kwh": 0.001,
                    "carbon_kg": 0.000233,
                }

        self.transformers.AutoTokenizer.from_pretrained = (
            lambda name: Tokenizer()
        )

        with (
            patch.object(
                self.module,
                "load_model_with_fallback",
                return_value=Model(),
            ),
            patch.object(
                self.module,
                "HybridEnergyMonitor",
                Monitor,
            ),
        ):
            result = self.module.run_single_experiment(
                {
                    "max_new_tokens": 2,
                    "warmup_runs": 0,
                },
                "fake",
                ["p"] * 5,
                4,
            )

        self.assertEqual(
            result["status"],
            "ok",
            result,
        )
        self.assertEqual(calls, [4, 1])
        self.assertEqual(result["total_output_tokens"], 10)
        self.assertEqual(result["total_input_tokens"], 10)
        self.assertEqual(result["num_batches"], 2)
        self.assertAlmostEqual(
            result["energy_per_token"],
            0.0001,
        )


class AggregationTests(unittest.TestCase):
    def setUp(self):
        fake_run = types.SimpleNamespace(
            run_single_experiment=lambda *a, **k: {}
        )

        fake_space = types.SimpleNamespace(
            BASELINE_CONFIG={},
            get_ofat_experiments=lambda: [],
        )

        with patch.dict(
            sys.modules,
            {
                "experiments.run_experiment": fake_run,
                "factors.search_space": fake_space,
                "yaml": types.SimpleNamespace(),
                "utils.seed": types.SimpleNamespace(
                    set_seed=lambda seed: None
                ),
            },
        ):
            self.module = load(
                "test_runner",
                "experiments/ofat_runner.py",
            )

    def test_raw_repetitions_survive_and_failures_are_excluded(self):
        keys = dict(
            configuration_id="baseline",
            experiment_type="baseline",
            factor="baseline",
            factor_value="Baseline",
            model="fake",
            quantization="fp32",
            batch_size=1,
            workload="short",
            hardware="cpu",
            measurement_method="software",
        )

        raw = pd.DataFrame(
            [
                dict(
                    keys,
                    status="ok",
                    repetition=1,
                    energy_per_token=2,
                ),
                dict(
                    keys,
                    status="ok",
                    repetition=2,
                    energy_per_token=4,
                ),
                dict(
                    keys,
                    status="failed",
                    repetition=3,
                ),
            ]
        )

        summary = self.module.summarize_results(raw)

        self.assertEqual(len(raw), 3)
        self.assertEqual(
            summary.iloc[0].energy_per_token,
            3,
        )
        self.assertEqual(
            summary.iloc[0].n_successful,
            2,
        )
        self.assertEqual(
            summary.iloc[0].n_attempted,
            3,
        )
        self.assertNotIn("repetition", summary)

        with tempfile.TemporaryDirectory(
            dir=ROOT / "tests"
        ) as folder:
            path = self.module.save_results(
                raw,
                "raw.csv",
                folder,
            )
            self.assertEqual(
                len(pd.read_csv(path)),
                3,
            )


if __name__ == "__main__":
    unittest.main()