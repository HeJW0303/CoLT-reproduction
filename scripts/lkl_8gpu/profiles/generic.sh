#!/usr/bin/env bash

# Generic single-node profile: require eight visible GPUs without constraining
# the vendor model name. Hardware-specific profiles remain optional.
COLT_EXPECTED_GPU_NAME=""
COLT_DEFAULT_EVAL_GPUS="0,1,2,3,4,5,6,7"
