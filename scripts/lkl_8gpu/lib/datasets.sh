#!/usr/bin/env bash

dataset_group() {
  case "$1" in
    chartqa|ChartQA_TEST) printf '%s\n' ChartQA_TEST ;;
    mmstar|MMStar) printf '%s\n' MMStar ;;
    phase1) printf '%s\n' ChartQA_TEST MMStar MMBench_DEV_EN ;;
    phase2) printf '%s\n' TextVQA_VAL MMT-Bench_VAL ;;
    phase3) printf '%s\n' SEEDBench_IMG ScienceQA_TEST AI2D_TEST ;;
    remaining7)
      printf '%s\n' ChartQA_TEST AI2D_TEST MMBench_DEV_EN ScienceQA_TEST TextVQA_VAL MMT-Bench_VAL SEEDBench_IMG
      ;;
    all|all8)
      printf '%s\n' ChartQA_TEST AI2D_TEST MMBench_DEV_EN ScienceQA_TEST TextVQA_VAL MMT-Bench_VAL SEEDBench_IMG MMStar
      ;;
    smoke) printf '%s\n' COLT_SMOKE_MMSTAR ;;
    SEEDBench_IMG|MMBench_DEV_EN|TextVQA_VAL|ScienceQA_TEST|AI2D_TEST|MMT-Bench_VAL)
      printf '%s\n' "$1"
      ;;
    *) return 1 ;;
  esac
}
dataset_url() {
  case "$1" in
    SEEDBench_IMG) echo 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv' ;;
    MMBench_DEV_EN) echo 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv' ;;
    ChartQA_TEST) echo 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv' ;;
    TextVQA_VAL) echo 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv' ;;
    ScienceQA_TEST) echo 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv' ;;
    MMStar) echo 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv' ;;
    AI2D_TEST) echo 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv' ;;
    MMT-Bench_VAL) echo 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv' ;;
    *) return 1 ;;
  esac
}

dataset_size() {
  case "$1" in
    SEEDBench_IMG) echo 619569378 ;; MMBench_DEV_EN) echo 37156625 ;;
    ChartQA_TEST) echo 129773428 ;; TextVQA_VAL) echo 1222773494 ;;
    ScienceQA_TEST) echo 51398059 ;; MMStar) echo 59552082 ;;
    AI2D_TEST) echo 167443652 ;; MMT-Bench_VAL) echo 631302456 ;;
    *) return 1 ;;
  esac
}

dataset_md5() {
  case "$1" in
    SEEDBench_IMG) echo 68017231464752261a2526d6ca3a10c0 ;;
    MMBench_DEV_EN) echo b6caf1133a01c6bb705cf753bb527ed8 ;;
    ChartQA_TEST) echo c902e0aa9be5582a7aad6dcf52734b42 ;;
    TextVQA_VAL) echo b233b31f551bbf4056f2f955da3a92cd ;;
    ScienceQA_TEST) echo e42e9e00f9c59a80d8a5db35bc32b71f ;;
    MMStar) echo e1ecd2140806c1b1bbf54b43372efb9e ;;
    AI2D_TEST) echo 0f593e0d1c7df9a3d69bf1f947e71975 ;;
    MMT-Bench_VAL) echo 8dd4b730f53dbf9c3aed90ca31c928e0 ;;
    *) return 1 ;;
  esac
}
