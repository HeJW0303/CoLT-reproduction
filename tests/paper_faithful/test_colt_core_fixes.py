import os

from scripts.lkl_8gpu.tools.verify_paper_faithful import (
    verify_aux_batching_switch,
    verify_backward_decoder_official_behavior,
    verify_backward_gradient_ownership,
    verify_empty_forward_targets,
    verify_forward_causal_alignment,
    verify_visual_partial_adaptation,
)


def test_forward_causal_alignment() -> None:
    verify_forward_causal_alignment()


def test_empty_forward_targets() -> None:
    verify_empty_forward_targets()


def test_visual_partial_adaptation() -> None:
    previous = os.environ.get("COLT_PAPER_FAITHFUL")
    os.environ["COLT_PAPER_FAITHFUL"] = "1"
    try:
        verify_visual_partial_adaptation()
    finally:
        if previous is None:
            os.environ.pop("COLT_PAPER_FAITHFUL", None)
        else:
            os.environ["COLT_PAPER_FAITHFUL"] = previous


def test_backward_gradient_ownership() -> None:
    verify_backward_gradient_ownership()


def test_backward_decoder_official_behavior() -> None:
    verify_backward_decoder_official_behavior()


def test_aux_batching_switch() -> None:
    verify_aux_batching_switch()
