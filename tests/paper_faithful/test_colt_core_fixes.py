import os

from scripts.lkl_8gpu.verify_paper_faithful import (
    verify_backward_gradient_ownership,
    verify_empty_forward_targets,
    verify_forward_causal_alignment,
    verify_visual_freeze_keys,
)


def test_forward_causal_alignment() -> None:
    verify_forward_causal_alignment()


def test_empty_forward_targets() -> None:
    verify_empty_forward_targets()


def test_visual_freeze_keys() -> None:
    previous = os.environ.get("COLT_PAPER_FAITHFUL")
    os.environ["COLT_PAPER_FAITHFUL"] = "1"
    try:
        verify_visual_freeze_keys()
    finally:
        if previous is None:
            os.environ.pop("COLT_PAPER_FAITHFUL", None)
        else:
            os.environ["COLT_PAPER_FAITHFUL"] = previous


def test_backward_gradient_ownership() -> None:
    verify_backward_gradient_ownership()
