#!/usr/bin/env bash
# SemArt smoke: every stage and method on one small 3-part scene with short training.
# Not a benchmark (2000 iters, res 2). Launch:
#   bash scripts/colab_job.sh scripts/jobs/semart_smoke.sh --gpu T4 --datasets "dta"
SCENES="dta/sapien/storage_47254"
METHODS="artgs,sem_k,sem_mask,sem_full"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=4                  # 25 views per state
ITERS=2000 RES=2 PREDICT_ITERS=500 SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=2000
source scripts/jobs/semart_lib.sh
