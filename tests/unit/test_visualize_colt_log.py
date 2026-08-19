import unittest

from scripts.lkl_8gpu.tools.visualize_colt_log import parse_trainer_and_components


class VisualizeColtLogTests(unittest.TestCase):
    def test_component_metrics_are_step_aligned_and_final_gap_is_explicit(self) -> None:
        text = """
***** Running training *****
ce_loss_total : 1.0
forward_loss_total : 2.0
backward_loss_total : 3.0
prediction_loss_total : 4.0
grounding_loss_total : 5.0
weighted_visual_term : 6.0
colt_control_rows : active=1 visual_cot=1 visual_only=0 image_mask_hit=0
{'loss': 7.0, 'grad_norm': 8.0, 'learning_rate': 0.001, 'epoch': 0.0}
{'loss': 6.0, 'grad_norm': 7.0, 'learning_rate': 0.0009, 'epoch': 0.1}
"""

        trainer, components, raw = parse_trainer_and_components(text)

        self.assertEqual(len(trainer), 2)
        self.assertEqual(len(components), 2)
        self.assertEqual(len(raw["grounding_loss_total"]), 1)
        self.assertTrue(bool(components.loc[0, "component_metrics_complete"]))
        self.assertTrue(bool(components.loc[0, "control_metrics_complete"]))
        self.assertFalse(bool(components.loc[1, "component_metrics_complete"]))
        self.assertEqual(int(components.loc[1, "component_record_count"]), 0)
        self.assertEqual(float(components.loc[0, "grounding_loss_total_mean"]), 5.0)
        self.assertEqual(float(components.loc[0, "control_visual_cot_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()
