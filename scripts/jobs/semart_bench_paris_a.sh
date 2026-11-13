#!/usr/bin/env bash
# SemArt v2 benchmark (paris_a): sem_v2_mask only, 3 seeds, 20k iters, res 2; 2 scene pipelines at once on one T4.
SCENES="paris/sapien/blade_103706 paris/sapien/foldchair_102255 paris/sapien/fridge_10905 paris/sapien/laptop_10211 paris/sapien/oven_101917"
METHODS="sem_v2_mask"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0 1 42" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
PARALLEL=1 MAXPAR=2
source scripts/jobs/semart_lib.sh
