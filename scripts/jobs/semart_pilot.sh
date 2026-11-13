#!/usr/bin/env bash
# SemArt pilot: full-length training (ArtGS schedule) on ONE scene, all methods, seed 0.
# One scene per job keeps a T4 job under the 5 CU cap. Override SCENES when launching a copy.
#   bash scripts/colab_job.sh scripts/jobs/semart_pilot.sh --gpu T4 --datasets "dta" --max-cu 5
SCENES="${SCENES:-dta/sapien/storage_47254}"
METHODS="artgs,sem_k,sem_vlm,sem_mask,sem_full"
SEGMENTER=sam3
VLM=Qwen/Qwen3-VL-4B-Instruct
VIEW_STRIDE=1                  # all 100 views per state (SAM3 fp16: ~0.6 s/view on T4)
ITERS=20000 RES=2 SEEDS="0" K_SOURCE=vlm LAMBDA_SEM=0.1 SEM_UNTIL=15000
source scripts/jobs/semart_lib.sh
