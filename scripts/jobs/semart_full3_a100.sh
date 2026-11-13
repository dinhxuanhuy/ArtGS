#!/usr/bin/env bash
# SemArt full comparison on 3 scenes in parallel on one A100: 5 methods, 20k iters, res 2, seed 0.
#   bash scripts/colab_job.sh scripts/jobs/semart_full3_a100.sh --gpu A100 --datasets "artgs dta" --max-cu 25 --max-hours 3
SCENES="artgs/sapien/storage_47648 artgs/sapien/storage_45503 dta/sapien/fridge_10489"
METHODS="artgs,sem_k,sem_vlm,sem_mask,sem_full"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
PARALLEL=1
source scripts/jobs/semart_lib.sh
