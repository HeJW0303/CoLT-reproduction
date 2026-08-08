from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "scripts/lkl_8gpu/easyr1/preflight.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("colt_easyr1_preflight", PREFLIGHT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_preflight_module()


class EasyR1PreflightTests(unittest.TestCase):
    def write(self, root: Path, relative_path: str, content: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def create_repo_fixture(self, root: Path) -> tuple[Path, Path]:
        repo_root = root / "repo"
        model_path = root / "model"
        self.write(
            repo_root,
            "transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
            "def latent_reasoning_generate(self):\n    pass\n",
        )
        self.write(
            model_path,
            "config.json",
            json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}),
        )
        self.write(
            model_path,
            "model.safetensors.index.json",
            json.dumps({"weight_map": {"alpha": "a.safetensors", "prj.1.weight": "a.safetensors"}}),
        )
        return repo_root, model_path

    def create_easyr1_fixture(self, root: Path, compatible: bool) -> Path:
        easyr1_root = root / "EasyR1"
        self.write(
            easyr1_root,
            "UPSTREAM.md",
            "- Upstream commit: `4a36ad286d04382fce9816ac4429e650157a5f11`\n",
        )
        if compatible:
            worker = (
                'if self.config.rollout.name == "colt_transformers":\n'
                "    self.rollout = CoLTTransformersRollout()\n"
                "    FSDPTransformersShardingManager()\n"
            )
            actor = "colt_rl_response_length = responses.size(-1)\n"
            dataset = "from .reasoning_prompt import format_reasoning_prompt\n"
            prompt = (
                "LATENT_REASONING_TEMPLATE = 'hidden reasoning; final answer only'\n"
                'if reasoning_mode == "latent":\n'
                "    pass\n"
            )
        else:
            worker = (
                "from .rollout import vLLMRollout\n"
                "self.rollout = vLLMRollout(\n"
                "FSDPVLLMShardingManager(\n"
            )
            actor = "output = self.actor_module(input_ids=input_ids)\n"
            dataset = (
                "Provide your thinking process between the <think> and </think> tags\n"
                "<answer><your_answer_here></answer>\n"
            )
            prompt = dataset
        self.write(easyr1_root, "verl/workers/fsdp_workers.py", worker)
        if compatible:
            self.write(
                easyr1_root,
                "verl/workers/rollout/colt_transformers_rollout.py",
                "class CoLTTransformersRollout:\n    pass\n",
            )
            self.write(
                easyr1_root,
                "verl/workers/sharding_manager/fsdp_transformers.py",
                "class FSDPTransformersShardingManager:\n    pass\n",
            )
            self.write(
                easyr1_root,
                "verl/reward_function/colt_outcome.py",
                "def parse_hidden_reasoning_answer(response):\n    return response, 1.0\n",
            )
            self.write(easyr1_root, "examples/colt_fixed_v2_outcome_grpo.yaml", "worker: {}\n")
        self.write(easyr1_root, "verl/workers/actor/dp_actor.py", actor)
        self.write(
            easyr1_root,
            "verl/workers/reward/function.py",
            'reward_tensor[i, cur_response_length - 1] = score["overall"]\n',
        )
        self.write(easyr1_root, "verl/utils/dataset.py", dataset)
        self.write(easyr1_root, "verl/utils/reasoning_prompt.py", prompt)
        self.write(easyr1_root, "verl/trainer/main.py", "def main():\n    pass\n")
        return easyr1_root

    def create_dataset(self, root: Path) -> Path:
        dataset_path = root / "train.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "problem": "Which option?",
                        "answer": "A",
                        "data_type": "image",
                        "problem_type": "multiple choice",
                        "problem_id": "sample-1",
                        "images": ["image.jpg"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return dataset_path

    def test_incompatible_onethinker_source_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=False)
            dataset_path = self.create_dataset(root)
            report = preflight.AuditReport()
            preflight.audit_easyr1(easyr1_root, report)
            preflight.audit_model(repo_root, model_path, report)
            preflight.audit_dataset(dataset_path, report)

        blocker_codes = {finding.code for finding in report.blockers}
        self.assertFalse(report.ready)
        self.assertEqual(
            blocker_codes,
            {
                "ROLLOUT_VLLM_ONLY",
                "COLT_OUTCOME_REWARD_MISSING",
                "COLT_RL_CONFIG_MISSING",
                "ACTOR_LOGPROB_NOT_LATENT_AWARE",
                "PROMPT_REQUIRES_VISIBLE_THINK",
            },
        )
        self.assertTrue(report.facts["outcome_reward_written_to_final_token"])
        self.assertEqual(report.facts["dataset_records"], 1)

    def test_compatible_contract_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=True)
            dataset_path = self.create_dataset(root)
            report = preflight.AuditReport()
            preflight.audit_easyr1(easyr1_root, report)
            preflight.audit_model(repo_root, model_path, report)
            preflight.audit_dataset(dataset_path, report)

        self.assertTrue(report.ready)
        self.assertEqual(report.blockers, [])
        self.assertEqual({warning.code for warning in report.warnings}, {"CHECKPOINT_NO_AUTO_MAP"})
        self.assertEqual(
            report.facts["easyr1_upstream_commit"],
            "4a36ad286d04382fce9816ac4429e650157a5f11",
        )

    def test_runtime_audit_names_missing_packages(self) -> None:
        report = preflight.AuditReport()
        original_modules = preflight.RUNTIME_MODULES
        preflight.RUNTIME_MODULES = {
            "json-package": "json",
            "definitely-missing-package": "colt_module_that_does_not_exist",
        }
        try:
            preflight.audit_runtime(report)
        finally:
            preflight.RUNTIME_MODULES = original_modules

        self.assertEqual(report.blockers[0].code, "RL_RUNTIME_DEPENDENCIES_MISSING")
        self.assertEqual(report.facts["runtime_missing_packages"], ["definitely-missing-package"])

    def test_dataset_schema_errors_name_every_missing_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "train.jsonl"
            dataset_path.write_text('{"problem": "p", "answer": "a"}\n', encoding="utf-8")
            report = preflight.AuditReport()
            preflight.audit_dataset(dataset_path, report)

        self.assertEqual(report.blockers[0].code, "RL_DATASET_SCHEMA_MISMATCH")
        for field in ("data_type", "problem_id", "problem_type"):
            self.assertIn(field, report.blockers[0].message)

    def test_cli_returns_two_for_blockers_and_json_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=False)
            dataset_path = self.create_dataset(root)
            command = [
                sys.executable,
                str(PREFLIGHT_PATH),
                "--repo-root",
                str(repo_root),
                "--easyr1-root",
                str(easyr1_root),
                "--model-path",
                str(model_path),
                "--train-file",
                str(dataset_path),
                "--json",
            ]
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        self.assertEqual(result.returncode, 2, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ready"])
        self.assertIn("ROLLOUT_VLLM_ONLY", {item["code"] for item in payload["blockers"]})

    def test_cli_allow_incomplete_preserves_report_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=False)
            missing_dataset = root / "missing.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREFLIGHT_PATH),
                    "--repo-root",
                    str(repo_root),
                    "--easyr1-root",
                    str(easyr1_root),
                    "--model-path",
                    str(model_path),
                    "--train-file",
                    str(missing_dataset),
                    "--allow-incomplete",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CoLT/EasyR1 readiness: BLOCKED", result.stdout)
        self.assertIn("RL_DATASET_MISSING", result.stdout)

    def test_unified_launcher_dispatches_rl_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=False)
            dataset_path = self.create_dataset(root)
            result = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts/lkl_8gpu/colt.sh"),
                    "rl",
                    "audit",
                    "--python",
                    sys.executable,
                    "--easyr1-root",
                    str(easyr1_root),
                    "--model-path",
                    str(model_path),
                    "--train-file",
                    str(dataset_path),
                    "--allow-incomplete",
                    "--json",
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ready"])
        self.assertIn("ROLLOUT_VLLM_ONLY", {item["code"] for item in payload["blockers"]})

    def test_unified_launcher_refuses_training_without_dataset(self) -> None:
        missing_dataset = REPO_ROOT / "tests" / "fixtures" / "missing-onethinker-rl.json"
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts/lkl_8gpu/colt.sh"),
                "rl",
                "train",
                "--train-file",
                str(missing_dataset),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing OneThinker RL dataset", result.stderr)

    def test_unified_launcher_train_dry_run_prints_local_easyr1_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, model_path = self.create_repo_fixture(root)
            easyr1_root = self.create_easyr1_fixture(root, compatible=True)
            dataset_path = self.create_dataset(root)
            fake_modules = root / "fake_modules"
            for module_name in set(preflight.RUNTIME_MODULES.values()):
                if module_name == "transformers":
                    continue
                self.write(fake_modules, f"{module_name}.py", "\n")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(fake_modules), str(REPO_ROOT / "transformers-4.57.0/src")]
            )
            result = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts/lkl_8gpu/colt.sh"),
                    "rl",
                    "train",
                    "--python",
                    sys.executable,
                    "--easyr1-root",
                    str(easyr1_root),
                    "--model-path",
                    str(model_path),
                    "--train-file",
                    str(dataset_path),
                    "--config",
                    str(REPO_ROOT / "EasyR1/examples/colt_fixed_v2_outcome_grpo.yaml"),
                    "--gpus",
                    "0,2",
                    "--max-steps",
                    "1",
                    "--dry-run",
                ],
                cwd=repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=0\\,2", result.stdout)
        self.assertIn("-m verl.trainer.main", result.stdout)
        self.assertIn(f"worker.actor.model.model_path={model_path}", result.stdout)
        self.assertIn("trainer.max_steps=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
