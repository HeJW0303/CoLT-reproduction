import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_paper_faithful_v2.sh"


class PaperFaithfulV2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_versions_all_persistent_artifacts(self) -> None:
        for expected in (
            "colt_paper_faithful_v1",
            "colt_paper_faithful_v2",
            'V1_EVAL_LOG_DIR="$LOG_ROOT/eval/paper-faithful-v1"',
            'V2_EVAL_LOG_DIR="$LOG_ROOT/eval/paper-faithful-v2"',
            "results-v1",
            "results-v2",
        ):
            self.assertIn(expected, self.source)
        self.assertIn("colt_paper_faithful_v1_train_", self.source)
        self.assertIn("paper-faithful-v1_", self.source)
        self.assertIn("COLT_EVAL_LOG_LABEL=paper-faithful-v2", self.source)
        self.assertNotIn("$LOG_ROOT/eval-v1", self.source)
        self.assertNotIn("$LOG_ROOT/eval-v2", self.source)

    def test_uses_current_b1_training_with_batching(self) -> None:
        self.assertIn("train paper-faithful", self.source)
        self.assertIn("--batch-aux", self.source)
        self.assertIn('save_steps=500', self.source)
        self.assertIn('seed=42', self.source)
        self.assertIn('data_seed=42', self.source)

    def test_verifies_before_comparable_all8_evaluation(self) -> None:
        verify = self.source.index("verify model paper-faithful")
        evaluate = self.source.index("eval paper-faithful all8")
        self.assertLess(verify, evaluate)
        for expected in (
            "--generation respect-args",
            "--empty-response-policy prevent",
            "--workers \"$WORKERS_PER_GPU\"",
            "--prefetch 1",
            "--empty-cache-every 0",
            "--reseed-per-sample 1",
        ):
            self.assertIn(expected, self.source)

    def test_archive_is_guarded_and_recoverable(self) -> None:
        self.assertIn("Both v1 source and destination exist", self.source)
        self.assertIn("Legacy paper-faithful checkpoint is not a complete", self.source)
        self.assertIn("--resume", self.source)
        self.assertNotIn("rm -rf", self.source)


if __name__ == "__main__":
    unittest.main()
