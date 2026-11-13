#!/usr/bin/env bash
# SemArt v3 benchmark (paris_2): method 'semart', 3 seeds, 20k iters, res 2; 2 scene pipelines at once on one T4.
SCENES="paris/sapien/oven_101917 paris/sapien/scissor_11100 paris/sapien/stapler_103111"
METHODS="semart"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1
ITERS=20000 RES=2 SEEDS="0 1 42" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
PARALLEL=1 MAXPAR=2
source scripts/jobs/semart_lib.sh
