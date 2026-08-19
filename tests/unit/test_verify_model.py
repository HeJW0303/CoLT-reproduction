import unittest

from scripts.lkl_8gpu.tools.verify_model import resolve_expected_step


class VerifyModelTests(unittest.TestCase):
    def test_defaults_to_trainer_max_steps(self):
        self.assertEqual(resolve_expected_step({"max_steps": 3465}, None), 3465)

    def test_explicit_step_overrides_trainer_metadata(self):
        self.assertEqual(resolve_expected_step({"max_steps": 3465}, 1910), 1910)

    def test_missing_or_invalid_max_steps_requires_explicit_step(self):
        for state in ({}, {"max_steps": 0}, {"max_steps": True}, {"max_steps": "3465"}):
            with self.subTest(state=state):
                with self.assertRaisesRegex(RuntimeError, "positive max_steps"):
                    resolve_expected_step(state, None)


if __name__ == "__main__":
    unittest.main()
