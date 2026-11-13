#!/usr/bin/env bash
# SemArt full comparison on one scene: 5 methods, 20k iters, res 2, seed 0 (same protocol as semart_pilot.sh).
#   bash scripts/colab_job.sh scripts/jobs/semart_full_storage_47648.sh --gpu T4 --datasets "artgs" --max-cu 5 --max-hours 4
SCENES="artgs/sapien/storage_47648"
METHODS="artgs,sem_k,sem_vlm,sem_mask,sem_full"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
source scripts/jobs/semart_lib.sh
