#!/usr/bin/env bash
# SemArt v2 benchmark (multi): sem_v2_mask only, 3 seeds, 20k iters, res 2; 2 scene pipelines at once on one T4.
SCENES="artgs/sapien/storage_47648 artgs/sapien/table_31249 artgs/sapien/table_25493 artgs/sapien/oven_101908 artgs/sapien/storage_45503"
METHODS="sem_v2_mask"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0 1 42" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
PARALLEL=1 MAXPAR=2
source scripts/jobs/semart_lib.sh
