import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_MODULE_ROOT = REPO_ROOT / "transformers-4.57.0/src/transformers/models/qwen3_vl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle_k = load_module("test_colt_oracle_k", QWEN_MODULE_ROOT / "oracle_k.py")
segmenter = load_module("test_colt_segment_teacher_blocks", REPO_ROOT / "scripts/oracle_k/segment_teacher_blocks.py")
packager = load_module("test_colt_package_oracle_dataset", REPO_ROOT / "scripts/oracle_k/package_oracle_dataset.py")
register_dataset = load_module(
    "test_colt_register_oracle_dataset", REPO_ROOT / "scripts/oracle_k/register_oracle_dataset.py"
)
modeling_oracle_k = (
    load_module("test_colt_modeling_oracle_k", QWEN_MODULE_ROOT / "modeling_oracle_k.py")
    if torch is not None
    else None
)


class DummyConfig:
    pass


class OracleKDatasetRegistrationTest(unittest.TestCase):
    def test_legacy_registration_is_upgraded_with_role_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "dataset_info.json"
            legacy_entry = {
                "file_name": "oracle/colt_sft_image_oracle_k.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
            }
            registry_path.write_text(
                json.dumps({"onethinker_sft_image_oracle_k": legacy_entry}),
                encoding="utf-8",
            )
            argv = [
                "register_oracle_dataset.py",
                "--dataset-info",
                str(registry_path),
                "--file-name",
                legacy_entry["file_name"],
            ]
            with patch.object(sys, "argv", argv):
                register_dataset.main()

            registered = json.loads(registry_path.read_text(encoding="utf-8"))[
                "onethinker_sft_image_oracle_k"
            ]
            self.assertEqual(
                registered["tags"],
                {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            )


class OracleKFormatTest(unittest.TestCase):
    def test_annotation_round_trip_preserves_assistant_content(self):
        content = "prefix<think>first\nsecond。third</think><answer>42</answer>suffix"
        segmented = "first\n<continue_think>second。<continue_think>third"
        annotated = oracle_k.annotate_assistant_content(content, segmented, max_k=8)

        parsed = oracle_k.parse_oracle_k_cot(oracle_k.get_assistant_cot(annotated), max_k=8)

        self.assertEqual(parsed.k, 3)
        self.assertEqual(parsed.blocks, ("first\n", "second。", "third"))
        self.assertEqual(oracle_k.remove_assistant_annotation(annotated, max_k=8), content)

    def test_declared_k_mismatch_is_rejected(self):
        with self.assertRaisesRegex(oracle_k.OracleKFormatError, "Declared Oracle K=3"):
            oracle_k.parse_oracle_k_cot(
                "<thought_segments>3</thought_segments>a<continue_think>b",
                max_k=8,
            )

    def test_teacher_text_change_is_rejected(self):
        content = "<think>abc</think><answer>x</answer>"
        with self.assertRaisesRegex(oracle_k.OracleKFormatError, "changed the original CoT"):
            oracle_k.annotate_assistant_content(content, "a<continue_think>bd", max_k=8)

    def test_k_above_max_is_rejected(self):
        segmented = oracle_k.CONTINUE_THINK.join(str(index) for index in range(9))
        with self.assertRaisesRegex(oracle_k.OracleKFormatError, "exceeds max_k=8"):
            oracle_k.annotate_segmented_cot(segmented, max_k=8)

    def test_k_one_works(self):
        annotated = oracle_k.annotate_segmented_cot("one complete block", max_k=8)
        parsed = oracle_k.parse_oracle_k_cot(annotated, max_k=8)
        self.assertEqual(parsed.k, 1)
        self.assertEqual(parsed.original_cot, "one complete block")


class OracleKSettingsTest(unittest.TestCase):
    def test_disabled_default_does_not_modify_baseline_config(self):
        config = DummyConfig()
        settings = oracle_k.resolve_oracle_k_settings(config, environ={})
        self.assertFalse(settings.enabled)
        self.assertFalse(hasattr(config, "colt_oracle_k_enabled"))

    def test_enabled_settings_persist_to_config(self):
        config = DummyConfig()
        settings = oracle_k.resolve_oracle_k_settings(
            config,
            environ={
                "COLT_ORACLE_K_ENABLED": "1",
                "COLT_ORACLE_K_MAX": "6",
                "COLT_ORACLE_K_BUDGET_CONDITIONING": "true",
            },
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(config.colt_oracle_k_max, 6)
        self.assertTrue(config.colt_oracle_k_budget_conditioning)

        with patch.dict(os.environ, {}, clear=True):
            reloaded = oracle_k.resolve_oracle_k_settings(config)
        self.assertTrue(reloaded.enabled)
        self.assertEqual(reloaded.max_k, 6)

    def test_predictor_settings_persist_to_config(self):
        config = DummyConfig()
        settings = oracle_k.resolve_oracle_k_settings(
            config,
            environ={
                "COLT_ORACLE_K_ENABLED": "1",
                "COLT_ORACLE_K_MAX": "8",
                "COLT_ORACLE_K_PREDICTOR_ENABLED": "1",
                "COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT": "0.35",
                "COLT_ORACLE_K_DYNAMIC_INFERENCE": "1",
            },
        )

        self.assertTrue(settings.predictor_enabled)
        self.assertAlmostEqual(settings.predictor_loss_weight, 0.35)
        self.assertTrue(settings.dynamic_inference)
        self.assertTrue(config.colt_oracle_k_predictor_enabled)
        self.assertAlmostEqual(config.colt_oracle_k_predictor_loss_weight, 0.35)
        self.assertTrue(config.colt_oracle_k_dynamic_inference)

        with patch.dict(os.environ, {}, clear=True):
            reloaded = oracle_k.resolve_oracle_k_settings(config)
        self.assertTrue(reloaded.predictor_enabled)
        self.assertAlmostEqual(reloaded.predictor_loss_weight, 0.35)
        self.assertTrue(reloaded.dynamic_inference)

    def test_dynamic_inference_requires_predictor(self):
        with self.assertRaisesRegex(ValueError, "DYNAMIC_INFERENCE requires"):
            oracle_k.resolve_oracle_k_settings(
                DummyConfig(),
                environ={
                    "COLT_ORACLE_K_ENABLED": "1",
                    "COLT_ORACLE_K_DYNAMIC_INFERENCE": "1",
                },
            )

    def test_predictor_requires_oracle_k(self):
        with self.assertRaisesRegex(ValueError, "PREDICTOR_ENABLED requires"):
            oracle_k.resolve_oracle_k_settings(
                DummyConfig(),
                environ={"COLT_ORACLE_K_PREDICTOR_ENABLED": "1"},
            )


class OracleKDistributedTrainingPlanTest(unittest.TestCase):
    def test_all_ranks_execute_the_global_maximum_number_of_calls(self):
        plans = [oracle_k.build_oracle_k_training_plan(local_k, 8) for local_k in (1, 3, 5, 8)]

        self.assertEqual({len(plan) for plan in plans}, {8})
        self.assertEqual(
            [[step.active for step in plan].count(True) for plan in plans],
            [1, 3, 5, 8],
        )
        self.assertTrue(all(sum(step.backward_index is not None for step in plan) == 7 for plan in plans))

    def test_dummy_steps_reuse_last_valid_local_indices(self):
        plan = oracle_k.build_oracle_k_training_plan(local_k=2, synchronized_k=5)

        self.assertEqual([step.forward_index for step in plan], [0, 1, 1, 1, 1])
        self.assertEqual([step.backward_index for step in plan], [None, 0, 1, 1, 1])
        self.assertEqual([step.active for step in plan], [True, True, False, False, False])

    def test_single_rank_plan_preserves_the_original_local_k_path(self):
        plan = oracle_k.build_oracle_k_training_plan(local_k=4, synchronized_k=4)

        self.assertTrue(all(step.active for step in plan))
        self.assertEqual([step.forward_index for step in plan], [0, 1, 2, 3])
        self.assertEqual([step.backward_index for step in plan], [None, 0, 1, 2])

    def test_synchronized_k_cannot_be_smaller_than_local_k(self):
        with self.assertRaisesRegex(ValueError, "synchronized_k must be at least local_k"):
            oracle_k.build_oracle_k_training_plan(local_k=3, synchronized_k=2)

    def test_predictor_loss_weight_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            oracle_k.resolve_oracle_k_settings(
                DummyConfig(),
                environ={
                    "COLT_ORACLE_K_ENABLED": "1",
                    "COLT_ORACLE_K_PREDICTOR_ENABLED": "1",
                    "COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT": "nan",
                },
            )

    def test_forced_inference_k_validation(self):
        self.assertEqual(oracle_k.resolve_forced_inference_k(8, {"COLT_INFERENCE_K": "5"}), 5)
        with self.assertRaisesRegex(ValueError, "must be in"):
            oracle_k.resolve_forced_inference_k(8, {"COLT_INFERENCE_K": "9"})

    def test_inference_k_priority(self):
        select = oracle_k.select_oracle_k_inference_steps
        self.assertEqual(
            select(8, 3, num_hidden_generations=6, forced_k=5, predicted_k=4, dynamic_inference=True),
            6,
        )
        self.assertEqual(select(8, 3, forced_k=5, predicted_k=4, dynamic_inference=True), 5)
        self.assertEqual(select(8, 3, predicted_k=4, dynamic_inference=True), 4)
        self.assertEqual(select(8, 3, predicted_k=4, dynamic_inference=False), 3)

    def test_explicit_k_does_not_depend_on_forced_environment_value(self):
        self.assertEqual(
            oracle_k.select_oracle_k_inference_steps(
                8,
                3,
                num_hidden_generations=6,
                forced_k=999,
                predicted_k=4,
                dynamic_inference=True,
            ),
            6,
        )

    def test_dynamic_inference_requires_prediction(self):
        with self.assertRaisesRegex(ValueError, "requires one predicted K"):
            oracle_k.select_oracle_k_inference_steps(8, 3, dynamic_inference=True)


class TeacherSegmentationTest(unittest.TestCase):
    def test_high_reasoning_effort_is_sent_to_teacher(self):
        import json

        captured_payload = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                response = {"choices": [{"message": {"content": '{"boundary_after_units":[1]}'}}]}
                return json.dumps(response).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured_payload.update(json.loads(request.data.decode("utf-8")))
            self.assertEqual(timeout, 30.0)
            return FakeResponse()

        with patch.object(segmenter.urllib.request, "urlopen", side_effect=fake_urlopen):
            boundaries = segmenter.request_segmentation(
                units=["Plan. ", "Solve."],
                text_context="[]",
                model="gpt-5.5",
                endpoint="https://example.invalid/v1/chat/completions",
                api_key="secret",
                min_k=1,
                max_k=8,
                timeout=30.0,
                max_output_tokens=1024,
                reasoning_effort="high",
                use_response_format=True,
            )

        self.assertEqual(boundaries, [1])
        self.assertEqual(captured_payload["reasoning_effort"], "high")
        self.assertEqual(captured_payload["max_tokens"], 1024)

    def test_prompt_requires_balanced_nonredundant_blocks(self):
        self.assertIn("Do not minimize or maximize K", segmenter.SYSTEM_PROMPT)
        self.assertIn("Do not merge distinct subgoals merely to reduce K", segmenter.SYSTEM_PROMPT)
        self.assertIn("answer-only conclusion", segmenter.SYSTEM_PROMPT)
        self.assertIn("Repetition is not a new stage", segmenter.SYSTEM_PROMPT)
        self.assertIn("target identification, box localization", segmenter.SYSTEM_PROMPT)
        self.assertIn("independent subcalculations or quantities", segmenter.SYSTEM_PROMPT)

    def test_unit_split_and_boundary_reconstruction_are_reversible(self):
        cot = "Plan first.\n\nDerive the result; then check it. Final answer"
        units = segmenter.split_cot_units(cot)

        segmented = segmenter.build_segmented_cot(units, [1, 3], min_k=1, max_k=8)

        self.assertEqual(segmented.replace("<continue_think>", ""), cot)
        self.assertEqual(segmented.count("<continue_think>"), 2)

    def test_teacher_receives_text_context_without_image_bytes(self):
        record = {
            "messages": [
                {"role": "user", "content": "<image>Which object is left?"},
                {"role": "assistant", "content": "<think>Inspect objects.</think><answer>cup</answer>"},
            ],
            "images": ["./large-image.jpg"],
        }

        context = segmenter.collect_text_context(record, assistant_index=1)

        self.assertIn("Which object is left?", context)
        self.assertNotIn("large-image.jpg", context)

    def test_chinese_punctuation_splits_without_following_spaces(self):
        cot = "先理解题意。然后列式计算；最后检查答案！"

        units = segmenter.split_cot_units(cot)

        self.assertEqual(units, ["先理解题意。", "然后列式计算；", "最后检查答案！"])
        self.assertEqual("".join(units), cot)

    def test_single_newlines_are_available_as_teacher_boundaries(self):
        cot = "Plan:\n1. derive equation\n2. verify result"

        units = segmenter.split_cot_units(cot)

        self.assertEqual(units, ["Plan:\n", "1. derive equation\n", "2. verify result"])
        self.assertEqual("".join(units), cot)

    def test_sentence_ending_in_digit_still_splits(self):
        cot = "The result is 5. Then verify."

        units = segmenter.split_cot_units(cot)

        self.assertEqual(units, ["The result is 5. ", "Then verify."])

    def test_invalid_teacher_boundaries_are_rejected(self):
        units = ["Plan. ", "Derive. ", "Check."]
        with self.assertRaisesRegex(segmenter.OracleKFormatError, "strictly increasing"):
            segmenter.build_segmented_cot(units, [2, 1], min_k=1, max_k=8)
        with self.assertRaisesRegex(segmenter.OracleKFormatError, "must be in"):
            segmenter.build_segmented_cot(units, [3], min_k=1, max_k=8)

    def test_segment_record_returns_compact_reversible_boundaries(self):
        record = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>plan. derive. check.</think><answer>42</answer>"},
            ],
            "images": ["./image.png"],
        }
        args = SimpleNamespace(
            model="teacher",
            endpoint="http://unused",
            api_key="",
            min_k=1,
            max_k=8,
            timeout=1.0,
            max_output_tokens=100,
            reasoning_effort="high",
            no_response_format=False,
            max_retries=0,
            retry_backoff=0.0,
        )
        boundaries = [1, 2]

        with patch.object(segmenter, "request_segmentation", return_value=boundaries):
            result = segmenter.segment_record(7, record, args)

        self.assertEqual(result["index"], 7)
        self.assertEqual(result["k"], 3)
        self.assertEqual(result["boundary_after_units"], boundaries)
        self.assertNotIn("annotated_content", result)

        _, content, cot = segmenter.find_assistant_message(record)
        segmented = segmenter.build_segmented_cot(
            segmenter.split_cot_units(cot),
            result["boundary_after_units"],
            min_k=1,
            max_k=8,
        )
        annotated = oracle_k.annotate_assistant_content(content, segmented, max_k=8)
        self.assertEqual(oracle_k.remove_assistant_annotation(annotated, max_k=8), content)


class OracleKPackagingTest(unittest.TestCase):
    def test_annotation_state_requires_exact_unique_range(self):
        import json
        import tempfile

        meta = {"type": "meta", "format_version": 5, "user_prompt_version": 3}
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "state.jsonl"
            state_path.write_text(
                "\n".join(
                    [
                        json.dumps(meta),
                        json.dumps({"type": "result", "index": 2}),
                        json.dumps({"type": "result", "index": 3}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded_meta, results = packager.load_annotation_state(state_path, {2, 3})

            self.assertEqual(loaded_meta, meta)
            self.assertEqual(set(results), {2, 3})

    def test_annotation_state_rejects_duplicate_results(self):
        import json
        import tempfile

        meta = {"type": "meta", "format_version": 5, "user_prompt_version": 3}
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "state.jsonl"
            state_path.write_text(
                "\n".join(
                    [
                        json.dumps(meta),
                        json.dumps({"type": "result", "index": 2}),
                        json.dumps({"type": "result", "index": 2}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate state result"):
                packager.load_annotation_state(state_path, {2})


@unittest.skipIf(torch is None, "torch is only installed in the CoLT training environment")
class OracleKConditionerTest(unittest.TestCase):
    def test_pooling_uses_last_valid_token_for_left_and_right_padding(self):
        hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        attention_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])

        pooled = modeling_oracle_k.pool_last_valid_hidden(hidden_states, attention_mask)

        self.assertTrue(torch.equal(pooled[0], hidden_states[0, 1]))
        self.assertTrue(torch.equal(pooled[1], hidden_states[1, 3]))

    def test_pooling_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            modeling_oracle_k.pool_last_valid_hidden(torch.zeros(1, 2, 3), torch.zeros(1, 2))

    def test_budget_and_step_embeddings_receive_gradients(self):
        conditioner = modeling_oracle_k.OracleKBudgetConditioner(max_k=8, hidden_size=4)
        latent = torch.zeros(1, 1, 4, requires_grad=True)

        conditioner(latent, oracle_k=3, step_index=2).sum().backward()

        self.assertGreater(conditioner.budget_embedding.weight.grad[3].abs().sum().item(), 0)
        self.assertGreater(conditioner.step_embedding.weight.grad[2].abs().sum().item(), 0)

    def test_k_predictor_outputs_all_classes_and_receives_gradients(self):
        predictor = modeling_oracle_k.OracleKPredictor(max_k=8, hidden_size=6)
        pooled_hidden = torch.randn(3, 6, requires_grad=True)

        logits = predictor(pooled_hidden)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3, 7]))
        loss.backward()

        self.assertEqual(logits.shape, (3, 8))
        self.assertGreater(predictor.network[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(pooled_hidden.grad.abs().sum().item(), 0)

    def test_k_predictor_rejects_unpooled_hidden_states(self):
        predictor = modeling_oracle_k.OracleKPredictor(max_k=8, hidden_size=6)
        with self.assertRaisesRegex(ValueError, "expects"):
            predictor(torch.zeros(2, 3, 6))


if __name__ == "__main__":
    unittest.main()
