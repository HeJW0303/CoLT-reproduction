from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "lkl_8gpu"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_paper_oracle_pipeline.sh"


class Lkl8GpuScriptTests(unittest.TestCase):
    def run_bash(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", script],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_all_shell_files_parse(self) -> None:
        shell_files = sorted(SCRIPT_ROOT.rglob("*.sh"))
        self.assertTrue(shell_files)
        result = subprocess.run(
            ["bash", "-n", *map(str, shell_files)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_has_no_profile_or_conda_side_effect(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT_ROOT / "colt.sh"), "help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Model path precedence", result.stdout)

    def test_dataset_groups_are_unique_and_complete(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/datasets.sh"; dataset_group all8'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        datasets = result.stdout.splitlines()
        self.assertEqual(len(datasets), 8)
        self.assertEqual(len(set(datasets)), 8)
        self.assertIn("ChartQA_TEST", datasets)
        self.assertIn("MMStar", datasets)

    def test_cli_model_path_overrides_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli_path = root / "cli"
            env_path = root / "env"
            cli_path.mkdir()
            env_path.mkdir()
            result = self.run_bash(
                f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
                f'source "{SCRIPT_ROOT}/lib/model.sh"; '
                f'COLT_EVAL_MODEL_PATH="{env_path}"; '
                f'resolve_eval_model official "{cli_path}"; '
                'printf "%s\n%s\n" "$EVAL_MODEL_PATH" "$EVAL_MODEL_PATH_SOURCE"'
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(cli_path.resolve()), "--model-path"])

    def test_environment_model_path_overrides_target_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory)
            result = self.run_bash(
                f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
                f'source "{SCRIPT_ROOT}/lib/model.sh"; '
                f'COLT_EVAL_MODEL_PATH="{env_path}"; '
                'resolve_eval_model codefaithful ""; '
                'printf "%s\n" "$EVAL_MODEL_PATH_SOURCE"'
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "COLT_EVAL_MODEL_PATH")

    def test_generation_modes_have_unambiguous_log_labels(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
            f'source "{SCRIPT_ROOT}/lib/model.sh"; '
            'generation_log_label official; '
            'generation_log_label respect-args; '
            'generation_log_label respect-args prevent'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["sampling_max256", "greedy_max8192", "greedy_max8192_prevent-empty"],
        )

    def test_root_contains_only_unified_shell_entry(self) -> None:
        root_shells = sorted(path.name for path in SCRIPT_ROOT.glob("*.sh"))
        self.assertEqual(root_shells, ["colt.sh"])

    def test_baseline_rejects_sampling_protocol_before_runtime_init(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT_ROOT / "colt.sh"),
                "eval",
                "baseline",
                "chartqa",
                "--generation",
                "official",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baseline only supports greedy + 8192", result.stderr)

    def test_baseline_rejects_empty_response_prevention(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT_ROOT / "colt.sh"),
                "eval",
                "baseline",
                "chartqa",
                "--empty-response-policy",
                "prevent",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baseline does not support", result.stderr)

    def test_pipeline_dry_run_generates_isolated_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conda_env = root / "conda" / "envs" / "colt"
            conda_bin = conda_env / "bin"
            conda_bin.mkdir(parents=True)
            (conda_bin / "python").symlink_to(sys.executable)
            conda_init = root / "conda" / "etc" / "profile.d" / "conda.sh"
            conda_init.parent.mkdir(parents=True)
            conda_init.write_text("# dry-run fixture\n", encoding="utf-8")

            base_model = root / "models" / "Qwen3-VL-8B-Instruct"
            decoder_model = root / "models" / "Qwen3-0.6B"
            for model in (base_model, decoder_model):
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}\n", encoding="utf-8")
                (model / "model.safetensors").touch()

            train_data = root / "train_data"
            train_data.mkdir()
            paper_data = train_data / "colt_sft_image.json"
            oracle_data = train_data / "colt_sft_image_oracle_k.json"
            paper_data.write_text("[]\n", encoding="utf-8")
            oracle_data.write_text("[]\n", encoding="utf-8")
            registry = {
                "onethinker_sft_image": {"file_name": paper_data.name},
                "onethinker_sft_image_oracle_k": {"file_name": oracle_data.name},
            }
            (train_data / "dataset_info.json").write_text(
                json.dumps(registry), encoding="utf-8"
            )

            run_dir = root / "pipeline_run"
            environment = os.environ.copy()
            environment.update(
                COLT_CONDA_INIT_SH=str(conda_init),
                COLT_CONDA_ENV_DIR=str(conda_env),
                COLT_BASE_MODEL_DIR=str(base_model),
                COLT_DECODER_MODEL_DIR=str(decoder_model),
                COLT_DATA_ROOT=str(train_data),
                COLT_ORACLE_K_DATA_FILE=str(oracle_data),
                COLT_EVAL_DATA_ROOT=str(root / "eval_data"),
                COLT_PIPELINE_ROOT=str(root / "pipeline_runs"),
                COLT_PIPELINE_CACHE_ROOT=str(root / "cache"),
                COLT_PIPELINE_RUN_DIR=str(run_dir),
                COLT_LKL_ROOT=str(root / "runtime"),
            )
            result = subprocess.run(
                ["bash", str(PIPELINE_SCRIPT), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            paper_config = yaml.safe_load(
                (run_dir / "configs" / "paper_faithful.yaml").read_text(encoding="utf-8")
            )
            oracle_config = yaml.safe_load(
                (run_dir / "configs" / "oracle_k.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(paper_config["model_name_or_path"], str(base_model))
            self.assertEqual(oracle_config["model_name_or_path"], str(base_model))
            self.assertEqual(paper_config["dataset"], "onethinker_sft_image")
            self.assertEqual(oracle_config["dataset"], "onethinker_sft_image_oracle_k")
            self.assertNotEqual(paper_config["output_dir"], oracle_config["output_dir"])
            self.assertTrue(Path(paper_config["deepspeed"]).is_absolute())
            self.assertNotIn("--workers", result.stdout)
            self.assertNotIn("--prefetch", result.stdout)
            self.assertIn("eval paper-faithful all8", result.stdout)
            self.assertIn("eval oracle-k all8", result.stdout)


if __name__ == "__main__":
    unittest.main()
