#!/usr/bin/env bash
# SemArt v2 check on one scene: v1 (sem_k) vs v2 (VLM+motion K, fused joint types, safer 2D loss), 20k iters.
SCENES="artgs/sapien/storage_45503"
METHODS="artgs,sem_k,sem_v2,sem_v2_mask"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
source scripts/jobs/semart_lib.sh
