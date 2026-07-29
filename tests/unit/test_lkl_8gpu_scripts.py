from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "lkl_8gpu"


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


if __name__ == "__main__":
    unittest.main()
