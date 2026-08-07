from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "scripts" / "lkl_8gpu" / "external_judge"
RUN_SCRIPT = PIPELINE_ROOT / "run.sh"
FIVE_MODEL_SCRIPT = PIPELINE_ROOT / "run_five_models.sh"
ORACLE_K_FIXED_K_SWEEP_SCRIPT = PIPELINE_ROOT / "run_oracle_k_fixed_k_sweep.sh"
VALIDATOR = PIPELINE_ROOT / "validate_results.py"
RESUME_PREPARER = PIPELINE_ROOT / "prepare_judge_resume.py"
MODEL_NAME = "Qwen3-VL-8B-Instruct-COLT"
EVAL_ID = "TEST_EVAL"
JUDGE_MODEL = "deepseek-v4-flash"


class ExternalJudgePipelineTests(unittest.TestCase):
    def test_mathverse_score_parser_accepts_explicit_binary_labels_only(self) -> None:
        vlmeval_root = str(REPO_ROOT / "Evaluation" / "VLMEvalKit")
        sys.path.insert(0, vlmeval_root)
        try:
            from vlmeval.dataset.utils.mathverse import parse_mathverse_score_response
        finally:
            sys.path.remove(vlmeval_root)

        self.assertEqual(parse_mathverse_score_response("0"), 0)
        self.assertEqual(parse_mathverse_score_response(" Judgement: 1\n"), 1)
        self.assertEqual(parse_mathverse_score_response("Judgment is 0."), 0)
        self.assertEqual(
            parse_mathverse_score_response("The answers differ.\nFinal Judgement: 0"), 0
        )
        self.assertIsNone(parse_mathverse_score_response("Judgement: 0\nJudgement: 1"))
        self.assertIsNone(parse_mathverse_score_response("The answer is probably one."))

    def test_dry_run_is_isolated_and_does_not_expose_key(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT), "eval", "official", "--dry-run", "--gpus", "4,5,6,7"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={"PATH": os.environ["PATH"]},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("deepseek-v4-flash", result.stdout)
        self.assertIn("reasoning_effort", result.stdout)
        self.assertIn("low", result.stdout)
        self.assertIn("eval/external_judge/results", result.stdout)
        self.assertIn("no API credentials were read", result.stdout)
        self.assertNotIn("OPENAI_API_KEY", result.stdout)

    def test_all_dry_run_does_not_download(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT), "all", "official", "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={"PATH": os.environ["PATH"]},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Dry run", result.stdout)
        self.assertNotIn("Downloading", result.stdout)

    def test_deepseek_loader_injects_credentials_only_into_child(self) -> None:
        loader = PIPELINE_ROOT / "load_codex_api.py"
        result = subprocess.run(
            [
                sys.executable,
                str(loader),
                "--provider",
                "deepseek",
                "--",
                sys.executable,
                "-c",
                (
                    "import os; "
                    "assert os.environ['OPENAI_API_KEY'] == 'sk-test'; "
                    "assert os.environ['OPENAI_API_BASE'] == "
                    "'https://api.deepseek.com/chat/completions'; "
                    "print('deepseek-child-ready')"
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={"PATH": os.environ["PATH"], "DEEPSEEK_API_KEY": "sk-test"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "deepseek-child-ready\n")
        self.assertNotIn("sk-test", result.stdout + result.stderr)

    def test_five_model_dry_run_is_explicit_serial_and_restartable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_paths = [root / name for name in ("code", "paper-v1", "paper-v2", "oracle", "base")]
            for path in model_paths:
                path.mkdir()
            conda_env = root / "conda"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            environment = os.environ.copy()
            environment.update(
                {
                    "COLT_CODEFAITHFUL_CHECKPOINT": str(model_paths[0]),
                    "COLT_PAPER_V1_CHECKPOINT": str(model_paths[1]),
                    "COLT_PAPER_V2_CHECKPOINT": str(model_paths[2]),
                    "COLT_ORACLE_K_CHECKPOINT": str(model_paths[3]),
                    "COLT_BASE_MODEL_DIR": str(model_paths[4]),
                    "COLT_CONDA_ENV_DIR": str(conda_env),
                    "COLT_FIVE_MODEL_RESULT_ROOT": str(root / "results"),
                    "COLT_FIVE_MODEL_LOG_ROOT": str(root / "logs"),
                }
            )
            result = subprocess.run(
                ["bash", str(FIVE_MODEL_SCRIPT), "--dry-run"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Evaluating "), 5)
        for label in ("codefaithful", "paper-faithful-v1", "paper-faithful-v2", "oracle-k", "qwen3-vl-8b-base"):
            self.assertIn(f"results/{label}", result.stdout)
        self.assertEqual(result.stdout.count("--generation respect-args"), 5)
        self.assertEqual(result.stdout.count("--latent-transition training-consistent"), 4)
        self.assertEqual(result.stdout.count("--empty-response-policy prevent"), 4)
        self.assertIn("--latent-transition official", result.stdout)
        self.assertIn("--empty-response-policy allow", result.stdout)
        self.assertNotIn("--no-reuse", result.stdout)
        self.assertIn("no download, model verification, inference, or judge call was started", result.stdout)

    def test_oracle_k_fixed_k_sweep_dry_run_is_isolated_and_uses_external_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "oracle-k"
            checkpoint.mkdir()
            conda_env = root / "conda"
            (conda_env / "bin").mkdir(parents=True)
            (conda_env / "bin" / "python").symlink_to(sys.executable)
            result = subprocess.run(
                [
                    "bash",
                    str(ORACLE_K_FIXED_K_SWEEP_SCRIPT),
                    "--k-values",
                    "1,3,8",
                    "--model-path",
                    str(checkpoint),
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
                env={**os.environ, "COLT_CONDA_ENV_DIR": str(conda_env)},
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("COLT_INFERENCE_K="), 3)
        for k in (1, 3, 8):
            self.assertIn(f"COLT_INFERENCE_K={k}", result.stdout)
            self.assertIn(f"results/k{k}", result.stdout)
            self.assertIn(f"logs/k{k}", result.stdout)
            self.assertIn(f"COLT_EVAL_LOG_LABEL=oracle-k-fixed-k{k}", result.stdout)
        self.assertIn("COLT_LOG_PREDICTED_K=1", result.stdout)
        self.assertIn("COLT_LOG_ORACLE_K_PLAN=1", result.stdout)
        self.assertIn("MathVista_MINI\\,MathVerse_MINI\\,MMVet", result.stdout)
        self.assertIn("--latent-transition training-consistent", result.stdout)
        self.assertIn("--api-nproc 8", result.stdout)
        self.assertIn("no download, model verification, inference, or judge call was started", result.stdout)
        self.assertNotIn("--no-reuse", result.stdout)

    def test_download_manifest_has_exact_integrity_metadata(self) -> None:
        script = REPO_ROOT / "scripts" / "lkl_8gpu" / "lib" / "datasets.sh"
        query = (
            f'source "{script}"; '
            "for d in MathVista_MINI MathVerse_MINI MMVet; do "
            'printf "%s|%s|%s|%s\\n" "$d" "$(dataset_url "$d")" '
            '"$(dataset_size "$d")" "$(dataset_md5 "$d")"; done'
        )
        result = subprocess.run(
            ["bash", "-c", query],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "MathVista_MINI|https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv|55136266|f199b98e178e5a2a20e7048f5dcb0464",
                "MathVerse_MINI|http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIV.tsv|155702395|5017caca32b7fa110c350a1bea861b65",
                "MMVet|https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv|42861244|748aa6d4aa9d4de798306a63718455e3",
            ],
        )

    def write_fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "LMUData"
        result_dir = root / "work" / MODEL_NAME / EVAL_ID
        data_root.mkdir()
        result_dir.mkdir(parents=True)
        indices = [1, 2]
        predictions = pd.DataFrame({"index": indices, "prediction": ["answer one", "answer two"]})

        for dataset in ("MathVista_MINI", "MathVerse_MINI", "MMVet"):
            pd.DataFrame({"index": indices, "question": ["q1", "q2"]}).to_csv(
                data_root / f"{dataset}.tsv", sep="\t", index=False
            )
            base = result_dir / f"{MODEL_NAME}_{dataset}.xlsx"
            predictions.to_excel(base, index=False)

        mathvista_base = result_dir / f"{MODEL_NAME}_MathVista_MINI"
        pd.DataFrame({"index": indices, "log": ["Succeed", "Prefetch succeed"]}).to_excel(
            Path(f"{mathvista_base}_{JUDGE_MODEL}.xlsx"), index=False
        )
        pd.to_pickle(
            {1: {"log": "Succeed", "res": "1"}, 2: {"log": "Prefetch succeed", "res": "2"}},
            Path(f"{mathvista_base}_{JUDGE_MODEL}.pkl"),
        )
        pd.DataFrame({"Task&Skill": ["Overall"], "tot": [2], "hit": [1], "acc": [50.0]}).to_csv(
            Path(f"{mathvista_base}_{JUDGE_MODEL}_score.csv"), index=False
        )

        mathverse_base = result_dir / f"{MODEL_NAME}_MathVerse_MINI"
        pd.DataFrame({"index": indices, "log_extract": ["Succeed", "Succeed"]}).to_excel(
            Path(f"{mathverse_base}_{JUDGE_MODEL}_extract.xlsx"), index=False
        )
        pd.to_pickle(
            {1: {"log_extract": "Succeed", "extract": "1"}, 2: {"log_extract": "Succeed", "extract": "2"}},
            Path(f"{mathverse_base}_{JUDGE_MODEL}_extract.pkl"),
        )
        pd.DataFrame(
            {
                "index": indices,
                "problem_version": ["Text Lite", "Vision Only"],
                "log_extract": ["Succeed", "Succeed"],
                "log_score": ["Succeed", "Prefetch succeed"],
                "score": [True, False],
            }
        ).to_excel(Path(f"{mathverse_base}_{JUDGE_MODEL}_score.xlsx"), index=False)
        pd.to_pickle(
            {1: {"log_score": "Succeed", "score": True}, 2: {"log_score": "Prefetch succeed", "score": False}},
            Path(f"{mathverse_base}_{JUDGE_MODEL}_score.pkl"),
        )
        pd.DataFrame(
            {"split": ["Text Lite", "Vision Only"], "Overall": [100.0, 0.0]}
        ).to_csv(
            Path(f"{mathverse_base}_{JUDGE_MODEL}_score.csv"), index=False
        )

        mmvet_base = result_dir / f"{MODEL_NAME}_MMVet"
        pd.DataFrame({"index": indices, "log": ["Succeed", "Succeed"], "score": [0.2, 0.8]}).to_excel(
            Path(f"{mmvet_base}_{JUDGE_MODEL}.xlsx"), index=False
        )
        pd.to_pickle(
            {1: {"log": "Succeed", "score": 0.2}, 2: {"log": "Succeed", "score": 0.8}},
            Path(f"{mmvet_base}_{JUDGE_MODEL}.pkl"),
        )
        pd.DataFrame({"Category": ["Overall"], "tot": [2], "acc": [50.0]}).to_csv(
            Path(f"{mmvet_base}_{JUDGE_MODEL}_score.csv"), index=False
        )
        pd.DataFrame({"Category": ["Overall"], "tot": [2], "acc": [50.0]}).to_csv(
            Path(f"{mmvet_base}_{JUDGE_MODEL}_score_fine.csv"), index=False
        )
        return data_root, root / "work"

    def run_validator(
        self, data_root: Path, work_dir: Path, allow_empty_predictions: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(VALIDATOR),
            "--work-dir",
            str(work_dir),
            "--model-name",
            MODEL_NAME,
            "--eval-id",
            EVAL_ID,
            "--data-root",
            str(data_root),
            "--judge-model",
            JUDGE_MODEL,
        ]
        if allow_empty_predictions:
            command.append("--allow-empty-predictions")
        command.extend(["MathVista_MINI", "MathVerse_MINI", "MMVet"])
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_validator_accepts_complete_results_and_continuous_mmvet_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root, work_dir = self.write_fixture(Path(directory))
            result = self.run_validator(data_root, work_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MathVista_MINI", result.stdout)
        self.assertIn("MathVerse_MINI", result.stdout)
        self.assertIn("MMVet", result.stdout)
        self.assertIn("50.000000", result.stdout)

    def test_validator_rejects_incomplete_judge_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            checkpoint = work_dir / MODEL_NAME / EVAL_ID / f"{MODEL_NAME}_MMVet_{JUDGE_MODEL}.pkl"
            pd.to_pickle({1: {"log": "Succeed", "score": 0.2}}, checkpoint)
            result = self.run_validator(data_root, work_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MMVet judge checkpoint index mismatch", result.stderr)

    def test_validator_allows_empty_predictions_only_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            prediction_file = work_dir / MODEL_NAME / EVAL_ID / f"{MODEL_NAME}_MathVista_MINI.xlsx"
            predictions = pd.read_excel(prediction_file)
            predictions.loc[0, "prediction"] = " "
            predictions.to_excel(prediction_file, index=False)
            rejected = self.run_validator(data_root, work_dir)
            accepted = self.run_validator(data_root, work_dir, allow_empty_predictions=True)

        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("prediction file contains 1 empty predictions", rejected.stderr)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_validator_rejects_mathverse_score_artifact_with_wrong_split_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            score_csv = (
                work_dir
                / MODEL_NAME
                / EVAL_ID
                / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.csv"
            )
            pd.DataFrame(
                {"split": ["Text Lite", "Vision Only"], "Overall": [0.0, 0.0]}
            ).to_csv(score_csv, index=False)
            result = self.run_validator(data_root, work_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MathVerse 'Text Lite' accuracy", result.stderr)

    def test_resume_preparer_retries_only_failed_mathverse_score_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            del data_root
            result_dir = work_dir / MODEL_NAME / EVAL_ID
            checkpoint = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.pkl"
            pd.to_pickle(
                {
                    1: {"log_score": "Succeed", "score": True},
                    2: {"log_score": "All 5 retries failed", "score": False},
                },
                checkpoint,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RESUME_PREPARER),
                    "--result-dir",
                    str(result_dir),
                    "--model-name",
                    MODEL_NAME,
                    "--judge-model",
                    JUDGE_MODEL,
                    "MathVerse_MINI",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            resumed = pd.read_pickle(checkpoint)
            score_file = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.xlsx"
            score_csv = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.csv"
            score_archives = list(result_dir.glob(f"{score_file.name}.failed-judge-*.bak"))
            score_csv_archives = list(result_dir.glob(f"{score_csv.name}.failed-judge-*.bak"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resumed, {1: {"log_score": "Succeed", "score": True}})
        self.assertFalse(score_file.exists())
        self.assertFalse(score_csv.exists())
        self.assertEqual(len(score_archives), 1)
        self.assertEqual(len(score_csv_archives), 1)
        self.assertIn("retry_records=1", result.stdout)

    def test_resume_preparer_recovers_labelled_legacy_mathverse_scores_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            del data_root
            result_dir = work_dir / MODEL_NAME / EVAL_ID
            checkpoint = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.pkl"
            pd.to_pickle(
                {
                    1: {"log_score": "All 5 retries failed", "score": False},
                    2: {
                        "log_score": (
                            "Try 0: output is ignored, res is Judgement: 1, failed to parse.\n"
                            "All 5 retries failed.\n"
                        ),
                        "score": False,
                    },
                },
                checkpoint,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RESUME_PREPARER),
                    "--result-dir",
                    str(result_dir),
                    "--model-name",
                    MODEL_NAME,
                    "--judge-model",
                    JUDGE_MODEL,
                    "MathVerse_MINI",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            resumed = pd.read_pickle(checkpoint)
            score_file = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.xlsx"
            score_csv = result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_score.csv"

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resumed[2]["score"], True)
        self.assertNotIn("All 5 retries failed", resumed[2]["log_score"])
        self.assertEqual(set(resumed), {2})
        self.assertFalse(score_file.exists())
        self.assertFalse(score_csv.exists())
        self.assertIn("Recovered 1 MathVerse score records", result.stdout)
        self.assertIn("retry_records=1", result.stdout)

    def test_resume_preparer_handles_mathvista_mathverse_extraction_and_mmvet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, work_dir = self.write_fixture(root)
            del data_root
            result_dir = work_dir / MODEL_NAME / EVAL_ID
            pd.to_pickle(
                {
                    1: {"log": "Succeed", "res": "1"},
                    2: {"log": "Failed to obtain answer via API", "res": ""},
                },
                result_dir / f"{MODEL_NAME}_MathVista_MINI_{JUDGE_MODEL}.pkl",
            )
            pd.to_pickle(
                {
                    1: {"log_extract": "Succeed", "extract": "1"},
                    2: {"log_extract": "All 5 retries failed", "extract": ""},
                },
                result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_extract.pkl",
            )
            pd.to_pickle(
                {
                    1: {"log": "Succeed", "score": 0.2},
                    2: {"log": "All 5 retries failed", "score": 0.0},
                },
                result_dir / f"{MODEL_NAME}_MMVet_{JUDGE_MODEL}.pkl",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RESUME_PREPARER),
                    "--result-dir",
                    str(result_dir),
                    "--model-name",
                    MODEL_NAME,
                    "--judge-model",
                    JUDGE_MODEL,
                    "MathVista_MINI",
                    "MathVerse_MINI",
                    "MMVet",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            mathvista = pd.read_pickle(result_dir / f"{MODEL_NAME}_MathVista_MINI_{JUDGE_MODEL}.pkl")
            mathverse = pd.read_pickle(
                result_dir / f"{MODEL_NAME}_MathVerse_MINI_{JUDGE_MODEL}_extract.pkl"
            )
            mmvet = pd.read_pickle(result_dir / f"{MODEL_NAME}_MMVet_{JUDGE_MODEL}.pkl")
            archived = list(result_dir.glob("*.failed-judge-*.bak"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(mathvista), {1})
        self.assertEqual(set(mathverse), {1})
        self.assertEqual(set(mmvet), {1})
        self.assertGreaterEqual(len(archived), 8)
        self.assertIn("retry_records=3", result.stdout)

    def test_external_judge_controller_has_separate_inference_and_judge_processes(self) -> None:
        script = (REPO_ROOT / "scripts" / "lkl_8gpu" / "commands" / "eval.sh").read_text()
        self.assertIn('torchrun "${args[@]}" --mode infer', script)
        self.assertIn('VLMEVAL_WORKERS_PER_GPU=1', script)
        self.assertIn('python run.py "${single_process_args[@]}"', script)
        self.assertIn('--mode eval --judge "$judge" --api-nproc "$judge_nproc"', script)
        self.assertIn('validation_args+=(--allow-empty-predictions)', script)


if __name__ == "__main__":
    unittest.main()
