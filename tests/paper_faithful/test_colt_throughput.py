from tests.integration.lkl_8gpu.verify_colt_throughput import (
    verify_backward_batch_equivalence,
    verify_decoder_check_runs_once,
    verify_forward_batch_equivalence,
    verify_hot_path_has_no_tensor_conditions_or_tensor_prints,
)


def test_decoder_check_runs_once() -> None:
    verify_decoder_check_runs_once()


def test_forward_batch_equivalence() -> None:
    verify_forward_batch_equivalence()


def test_backward_batch_equivalence() -> None:
    verify_backward_batch_equivalence()


def test_hot_path_has_no_tensor_conditions_or_tensor_prints() -> None:
    verify_hot_path_has_no_tensor_conditions_or_tensor_prints()
