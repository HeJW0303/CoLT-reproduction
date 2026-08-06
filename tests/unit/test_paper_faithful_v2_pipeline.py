import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_paper_faithful_v2.sh"


class PaperFaithfulV2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_has_one_editable_runtime_block(self) -> None:
        for expected in (
            "BASE_MODEL_PATH=",
            "DECODER_MODEL_PATH=",
            "TRAIN_DATASET=",
            "TRAIN_DATASET_DIR=",
            "TRAIN_MEDIA_DIR=",
            "EVAL_DATASET_GROUP=",
            "EVAL_DATA_ROOT=",
            "WORKSPACE_ROOT=",
            "TRAIN_OUTPUT_DIR=",
            "EVAL_MODEL_PATH=",
            "TRAIN_GPUS=",
            "EVAL_GPUS=",
        ):
            self.assertIn(expected, self.source)

    def test_reuses_active_terminal_environment(self) -> None:
        self.assertIn('${CONDA_PREFIX:-}', self.source)
        self.assertIn('${VIRTUAL_ENV:-}', self.source)
        self.assertIn("command -v python", self.source)
        self.assertIn("conda activate <your-prepared-environment>", self.source)
        self.assertNotIn('source "$CONDA_INIT_SH"', self.source)
        self.assertNotIn("COLT_CONDA_ENV_DIR", self.source)

    def test_first_run_has_no_v1_or_fixed_step_dependency(self) -> None:
        for removed in (
            "V1_CHECKPOINT",
            "archive_path",
            "archive_training_logs",
            "EXPECTED_STEPS",
            "model_is_complete",
            "1910",
        ):
            self.assertNotIn(removed, self.source)

    def test_does_not_check_gpu_availability(self) -> None:
        for removed in (
            "nvidia-smi",
            "COLT_CHECK_GPU_FREE",
            "COLT_STRICT_PREFLIGHT",
            "Exactly eight training GPUs",
        ):
            self.assertNotIn(removed, self.source)

    def test_runs_training_then_selected_evaluation(self) -> None:
        train = self.source.index("train paper-faithful")
        evaluate = self.source.index('eval paper-faithful "$EVAL_DATASET_GROUP"')
        self.assertLess(train, evaluate)
        self.assertNotIn("verify model paper-faithful", self.source)
        self.assertIn("read_trained_model_step", self.source)
        for expected in (
            "--batch-aux",
            "--generation respect-args",
            "--empty-response-policy prevent",
            '--workers "$EVAL_WORKERS_PER_GPU"',
            "--prefetch 1",
            "--empty-cache-every 0",
            "--reseed-per-sample 1",
        ):
            self.assertIn(expected, self.source)

    def test_generated_config_contains_user_paths(self) -> None:
        for expected in (
            "model_name_or_path=base_model",
            "dataset=dataset",
            "dataset_dir=dataset_dir",
            "media_dir=media_dir",
            "output_dir=output_dir",
            "paper_faithful_v2_tokenized",
        ):
            self.assertIn(expected, self.source)

    def test_dry_run_uses_configured_paths_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory) / "runtime"
            environment = os.environ.copy()
            environment.update(
                CONDA_PREFIX=str(Path(sys.executable).parents[1]),
                COLT_PYTHON=sys.executable,
                COLT_LKL_ROOT=directory,
                COLT_RUNTIME_ROOT=str(runtime_root),
                COLT_BASE_MODEL_DIR="/models/base",
                COLT_DECODER_MODEL_DIR="/models/decoder",
                COLT_DATA_ROOT="/datasets/train",
                COLT_TRAIN_MEDIA_DIR="/datasets/media",
                COLT_TRAIN_DATASET="custom_train",
                COLT_EVAL_DATA_ROOT="/datasets/eval",
                COLT_EVAL_GROUP="TextVQA_VAL",
                COLT_EVAL_GPUS="4,5",
            )
            result = subprocess.run(
                ["bash", str(SCRIPT), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            generated_config = (
                runtime_root / "logs/paper_faithful_v2/paper_faithful_v2.yaml"
            )
            config = yaml.safe_load(generated_config.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train paper-faithful", result.stdout)
        self.assertIn("eval paper-faithful TextVQA_VAL", result.stdout)
        self.assertIn("--gpus 4\\,5", result.stdout)
        self.assertEqual(config["model_name_or_path"], "/models/base")
        self.assertEqual(config["dataset"], "custom_train")
        self.assertEqual(config["dataset_dir"], "/datasets/train")
        self.assertEqual(config["media_dir"], "/datasets/media")
        self.assertEqual(
            config["output_dir"],
            str(runtime_root / "checkpoints/colt_paper_faithful_v2"),
        )


if __name__ == "__main__":
    unittest.main()
