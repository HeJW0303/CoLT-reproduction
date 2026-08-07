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
TRANSITION_EVAL_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "lkl_8gpu"
    / "20_eval_transition_consistency_4checkpoints.sh"
)
ORACLE_K_SWEEP_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "lkl_8gpu"
    / "21_eval_oracle_k_sweep_chart_text.sh"
)
DECOUPLED_STEP_SWEEP_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "lkl_8gpu"
    / "22_eval_decoupled_steps_chart_text.sh"
)
SFT_WORKFLOW = (
    REPO_ROOT
    / "LLaMA-Factory"
    / "src"
    / "llamafactory"
    / "train"
    / "sft"
    / "workflow.py"
)


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
        shell_files = sorted(SCRIPT_ROOT.rglob("*.sh")) + [
            PIPELINE_SCRIPT,
            TRANSITION_EVAL_SCRIPT,
            ORACLE_K_SWEEP_SCRIPT,
            DECOUPLED_STEP_SWEEP_SCRIPT,
        ]
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

    def test_chart_text_dataset_group_contains_only_target_tasks(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/datasets.sh"; dataset_group chart-text'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["ChartQA_TEST", "TextVQA_VAL"])

    def test_oracle_k_sweep_dry_run_is_isolated_and_eval_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "oracle-k-checkpoint"
            checkpoint.mkdir()
            conda_env = root / "conda-env"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            result = subprocess.run(
                [
                    "bash",
                    str(ORACLE_K_SWEEP_SCRIPT),
                    "--k-values",
                    "1,4,8",
                    "--model-path",
                    str(checkpoint),
                    "--conda-env",
                    str(conda_env),
                    "--output-root",
                    str(root / "results"),
                    "--log-root",
                    str(root / "logs"),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("COLT_INFERENCE_K="), 3)
        for k in (1, 4, 8):
            self.assertIn(f"COLT_INFERENCE_K={k}", result.stdout)
            self.assertIn(f"results/k{k}", result.stdout)
            self.assertIn(f"logs/k{k}", result.stdout)
            self.assertIn(f"COLT_EVAL_LOG_LABEL=oracle-k-fixed-k{k}", result.stdout)
        self.assertIn("COLT_LOG_PREDICTED_K=1", result.stdout)
        self.assertIn("eval oracle-k chart-text", result.stdout)
        self.assertIn("--generation respect-args", result.stdout)
        self.assertIn("--latent-transition official", result.stdout)
        self.assertIn("--empty-response-policy prevent", result.stdout)
        self.assertNotIn(" train ", result.stdout)
        self.assertNotIn("--no-reuse", result.stdout)

    def test_oracle_k_sweep_rejects_out_of_range_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle-k-checkpoint"
            checkpoint.mkdir()
            conda_env = Path(directory) / "conda-env"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            result = subprocess.run(
                [
                    "bash",
                    str(ORACLE_K_SWEEP_SCRIPT),
                    "--k-values",
                    "0,9",
                    "--model-path",
                    str(checkpoint),
                    "--conda-env",
                    str(conda_env),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Each K must be an integer in [1, 8]", result.stderr)

    def test_decoupled_step_sweep_dry_run_is_isolated_and_eval_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "oracle-k-checkpoint"
            checkpoint.mkdir()
            conda_env = root / "conda-env"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            environment = os.environ.copy()
            environment["COLT_INFERENCE_K"] = "7"
            result = subprocess.run(
                [
                    "bash",
                    str(DECOUPLED_STEP_SWEEP_SCRIPT),
                    "--step-values",
                    "1,3,8",
                    "--model-path",
                    str(checkpoint),
                    "--conda-env",
                    str(conda_env),
                    "--output-root",
                    str(root / "results"),
                    "--log-root",
                    str(root / "logs"),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("COLT_INFERENCE_TRANSITION_STEPS="), 3)
        for steps in (1, 3, 8):
            self.assertIn(f"COLT_INFERENCE_TRANSITION_STEPS={steps}", result.stdout)
            self.assertIn(f"results/s{steps}", result.stdout)
            self.assertIn(f"logs/s{steps}", result.stdout)
            self.assertIn(
                f"COLT_EVAL_LOG_LABEL=oracle-k-predicted-budget-s{steps}",
                result.stdout,
            )
        self.assertNotIn("COLT_INFERENCE_K=", result.stdout)
        self.assertIn("COLT_LOG_PREDICTED_K=1", result.stdout)
        self.assertIn("COLT_LOG_ORACLE_K_PLAN=1", result.stdout)
        self.assertIn("COLT_EVAL_SEED=1234", result.stdout)
        self.assertIn("COLT_EVAL_JUDGE=exact_matching", result.stdout)
        self.assertIn("COLT_EVAL_RESULT_KIND=standard", result.stdout)
        self.assertIn("eval oracle-k chart-text", result.stdout)
        self.assertIn("--generation respect-args", result.stdout)
        self.assertIn("--latent-transition official", result.stdout)
        self.assertIn("--empty-response-policy prevent", result.stdout)
        self.assertNotIn(" train ", result.stdout)
        self.assertNotIn("--no-reuse", result.stdout)

    def test_decoupled_step_sweep_rejects_out_of_range_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "oracle-k-checkpoint"
            checkpoint.mkdir()
            conda_env = root / "conda-env"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            result = subprocess.run(
                [
                    "bash",
                    str(DECOUPLED_STEP_SWEEP_SCRIPT),
                    "--step-values",
                    "0,9",
                    "--model-path",
                    str(checkpoint),
                    "--conda-env",
                    str(conda_env),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("transition-step count must be an integer in [1, 8]", result.stderr)

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

    def test_eval_log_label_only_changes_log_path(self) -> None:
        source = (SCRIPT_ROOT / "commands" / "eval.sh").read_text(encoding="utf-8")
        self.assertIn('log_label="${COLT_EVAL_LOG_LABEL:-$target}"', source)
        self.assertIn('log_dir="$EVAL_LOG_ROOT/$log_label"', source)
        self.assertIn(
            'log_file="$log_dir/${log_label}_${group}_${generation_label}_${run_id}.log"',
            source,
        )
        self.assertIn('work_dir="$EVAL_OUTPUT_ROOT/$target/$group/$profile"', source)
        self.assertIn('eval_id="$(printf \'%s_%s\' "$target" "$profile"', source)

    def test_eval_rejects_unsafe_log_label_before_runtime_init(self) -> None:
        environment = os.environ.copy()
        environment["COLT_EVAL_LOG_LABEL"] = "../outside"
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT_ROOT / "colt.sh"),
                "eval",
                "paper-faithful",
                "chartqa",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("COLT_EVAL_LOG_LABEL must contain only", result.stderr)

    def test_eval_rejects_unknown_latent_transition_before_runtime_init(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT_ROOT / "colt.sh"),
                "eval",
                "paper-faithful",
                "chartqa",
                "--latent-transition",
                "unknown",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Latent transition must be official or training-consistent", result.stderr)

    def test_transition_mode_is_fingerprinted_and_forces_runtime_overlay(self) -> None:
        source = (SCRIPT_ROOT / "commands" / "eval.sh").read_text(encoding="utf-8")
        self.assertIn('--setting "latent_transition=$latent_transition"', source)
        self.assertIn('_lt${latent_transition//-/_}_seed', source)
        self.assertIn('"$latent_transition" != official', source)
        self.assertIn('export COLT_INFERENCE_LATENT_TRANSITION="$latent_transition"', source)

        model_source = (
            REPO_ROOT
            / "transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py"
        ).read_text(encoding="utf-8")
        self.assertIn("initialize_colt_inference_latent(", model_source)
        self.assertIn("advance_colt_inference_latent(", model_source)

    def test_four_checkpoint_transition_script_is_explicit_and_restartable(self) -> None:
        source = TRANSITION_EVAL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("colt_codefaithful", source)
        self.assertIn("colt_paper_faithful_v1", source)
        self.assertIn("colt_paper_faithful_v2", source)
        self.assertIn("colt_oracle_k_predictor_batch_285190c", source)
        self.assertEqual(source.count("--latent-transition training-consistent"), 1)
        self.assertNotIn("--no-reuse", source)
        self.assertIn("unset COLT_INFERENCE_K", source)
        self.assertLess(source.index("Preflighting all checkpoints"), source.index("Evaluating $label"))

    def test_tracked_root_contains_only_unified_shell_entry(self) -> None:
        pathspec = f":(glob){SCRIPT_ROOT.relative_to(REPO_ROOT)}/*.sh"
        result = subprocess.run(
            ["git", "ls-files", pathspec],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        root_shells = sorted(Path(path).name for path in result.stdout.splitlines())
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

            base_model = root / "models" / "Qwen3-VL-8B-Instruct"
            decoder_model = root / "models" / "Qwen3-0.6B"
            for model in (base_model, decoder_model):
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}\n", encoding="utf-8")
                (model / "model.safetensors").touch()

            train_data = root / "train_data"
            train_data.mkdir()
            paper_data = train_data / "colt_sft_image.json"
            oracle_data = train_data / "oracle_nested" / "colt_sft_image_oracle_k.json"
            oracle_data.parent.mkdir()
            paper_data.write_text("[]\n", encoding="utf-8")
            oracle_data.write_text("[]\n", encoding="utf-8")
            registry = {
                "onethinker_sft_image": {"file_name": paper_data.name},
                "onethinker_sft_image_oracle_k": {
                    "file_name": str(oracle_data.relative_to(train_data))
                },
            }
            (train_data / "dataset_info.json").write_text(
                json.dumps(registry), encoding="utf-8"
            )

            run_dir = root / "pipeline_run"
            environment = os.environ.copy()
            environment.update(
                CONDA_PREFIX=str(conda_env),
                CONDA_DEFAULT_ENV="colt",
                PATH=f"{conda_bin}{os.pathsep}{environment['PATH']}",
                COLT_BASE_MODEL_DIR=str(base_model),
                COLT_DECODER_MODEL_DIR=str(decoder_model),
                COLT_DATA_ROOT=str(train_data),
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
            self.assertTrue(paper_config["deepspeed"].endswith("ds_z3_8gpu.json"))
            self.assertNotIn("--workers", result.stdout)
            self.assertNotIn("--prefetch", result.stdout)
            self.assertIn("eval paper-faithful all8", result.stdout)
            self.assertIn("eval oracle-k all8", result.stdout)
            environment_text = (run_dir / "pipeline_environment.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn(f"oracle_data={oracle_data.resolve()}", environment_text)
            self.assertIn(f"python={conda_bin / 'python'}", environment_text)
            self.assertIn(f"active_environment={conda_env}", environment_text)

    def test_runtime_reuses_the_active_conda_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conda_env = Path(directory) / "friend_env"
            conda_bin = conda_env / "bin"
            conda_bin.mkdir(parents=True)
            (conda_bin / "python").symlink_to(sys.executable)
            environment = os.environ.copy()
            environment.update(
                CONDA_PREFIX=str(conda_env),
                CONDA_DEFAULT_ENV="friend_env",
                PATH=f"{conda_bin}{os.pathsep}/usr/bin:/bin",
                COLT_CONDA_INIT_SH="/path/that/does/not/exist/conda.sh",
                COLT_CONDA_ENV_DIR="/path/that/does/not/exist/env",
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    f'source "{SCRIPT_ROOT}/lib/runtime.sh"; activate_colt_env; command -v python',
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Using active Conda environment: friend_env", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], str(conda_bin / "python"))

    def test_gpu_idle_check_is_opt_in(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
            f'source "{SCRIPT_ROOT}/lib/gpu.sh"; '
            'COLT_GPU_IDS=(0 1); '
            'nvidia-smi() { echo "unexpected nvidia-smi call" >&2; return 99; }; '
            'maybe_check_selected_gpus_free'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU idle-memory check: skipped", result.stdout)

    def test_strict_preflight_enables_gpu_idle_check(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
            f'source "{SCRIPT_ROOT}/lib/gpu.sh"; '
            'COLT_STRICT_PREFLIGHT=1; COLT_GPU_IDS=(0); '
            'nvidia-smi() { printf "999\n"; }; '
            'maybe_check_selected_gpus_free'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Physical GPU 0 is not free", result.stderr)

    def test_pipeline_does_not_require_conda_install_paths(self) -> None:
        source = PIPELINE_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("COLT_CONDA_INIT_SH", source)
        self.assertNotIn("COLT_CONDA_ENV_DIR", source)
        self.assertIn('PYTHON_BIN="${COLT_PYTHON:-$(command -v python || true)}"', source)

    def test_pipeline_smoke_uses_short_temp_path_and_single_preprocessor(self) -> None:
        source = PIPELINE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("/tmp/colt_pipeline_$run_key", source)
        self.assertIn("preprocessing_num_workers=1", source)
        self.assertIn('export COLT_TMP_ROOT="$PIPELINE_TMP_ROOT"', source)

    def test_evaluation_acceleration_is_enabled_by_default(self) -> None:
        eval_source = (SCRIPT_ROOT / "commands" / "eval.sh").read_text(encoding="utf-8")
        pipeline_source = PIPELINE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('workers="${VLMEVAL_WORKERS_PER_GPU:-3}"', eval_source)
        self.assertIn('prefetch="${VLMEVAL_PREFETCH:-1}"', eval_source)
        self.assertIn('empty_cache="${VLMEVAL_EMPTY_CACHE_EVERY_N:-0}"', eval_source)
        self.assertNotIn("--workers", pipeline_source.split("evaluate_target()", 1)[1])
        self.assertNotIn("--prefetch", pipeline_source.split("evaluate_target()", 1)[1])

    def test_generic_profile_accepts_any_eight_gpu_models(self) -> None:
        result = self.run_bash(
            f'source "{SCRIPT_ROOT}/lib/runtime.sh"; '
            f'source "{SCRIPT_ROOT}/lib/gpu.sh"; '
            'COLT_GPU_PROFILE=generic; '
            'load_gpu_profile; '
            'nvidia-smi() { '
            'if [[ "$*" == *"--query-gpu=name"* ]]; then '
            'printf "NVIDIA H20\\n%.0s" {1..8}; '
            'else return 1; fi; }; '
            'validate_gpu_profile; '
            'printf "%s\\n%s\\n" "$COLT_GPU_PROFILE" "$COLT_DEFAULT_EVAL_GPUS"'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["generic", "0,1,2,3,4,5,6,7"])

    def test_pipeline_does_not_require_a_gpu_model_profile(self) -> None:
        source = PIPELINE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('GPU_PROFILE="${COLT_GPU_PROFILE:-generic}"', source)
        self.assertNotIn("COLT_GPU_PROFILE must be a100 or a800", source)

    def test_non_oracle_training_disables_dynamic_k(self) -> None:
        train_source = (SCRIPT_ROOT / "commands" / "train.sh").read_text(encoding="utf-8")
        self.assertEqual(train_source.count("export COLT_ORACLE_K_DYNAMIC_INFERENCE=0"), 2)

        pipeline_source = PIPELINE_SCRIPT.read_text(encoding="utf-8")
        paper_stage = pipeline_source.index("run_stage paper_train")
        dynamic_k = pipeline_source.index("export COLT_ORACLE_K_DYNAMIC_INFERENCE=1")
        oracle_stage = pipeline_source.index("run_stage oracle_train")
        self.assertLess(paper_stage, dynamic_k)
        self.assertLess(dynamic_k, oracle_stage)

    def test_oracle_k_checkpoint_and_aux_chunk_policy(self) -> None:
        config_path = (
            REPO_ROOT
            / "LLaMA-Factory"
            / "examples"
            / "train_full"
            / "colt_qwen3_sft_lkl_8gpu_oracle_k_predictor.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["save_steps"], 500)
        self.assertEqual(config["save_total_limit"], 1)

        train_source = (SCRIPT_ROOT / "commands" / "train.sh").read_text(encoding="utf-8")
        self.assertIn('COLT_AUX_MAX_BATCH_TOKENS="${COLT_AUX_MAX_BATCH_TOKENS:-4096}"', train_source)
        self.assertIn("COLT_AUX_MAX_BATCH_TOKENS must be a positive integer", train_source)

    def test_oracle_k_preflight_runs_before_trainer_construction(self) -> None:
        source = SFT_WORKFLOW.read_text(encoding="utf-8")
        model_load = source.index("model = load_model(")
        preflight = source.index("model.prepare_oracle_k_predictor_for_training()")
        trainer_construction = source.index("trainer = CustomSeq2SeqTrainer(")

        self.assertLess(model_load, preflight)
        self.assertLess(preflight, trainer_construction)
        self.assertIn("training_args.do_train", source[model_load:preflight])

    def test_oracle_k_fresh_and_resume_initialization_guards(self) -> None:
        source = (SCRIPT_ROOT / "commands" / "train.sh").read_text(encoding="utf-8")

        self.assertIn(
            'COLT_ORACLE_K_INITIALIZE_PREDICTOR:-$((1 - resume))', source
        )
        self.assertIn(
            'COLT_ORACLE_K_REQUIRE_PREOPTIMIZER_INIT:-$((1 - resume))', source
        )
        self.assertIn(
            '[[ "$resume" == 0 || "$COLT_ORACLE_K_INITIALIZE_PREDICTOR" == 0 ]]',
            source,
        )


if __name__ == "__main__":
    unittest.main()
