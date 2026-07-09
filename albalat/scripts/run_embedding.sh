#!/bin/bash

N_GPUS=$SLURM_GPUS_ON_NODE
source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc-per-node=N_GPUS -p embedding.nproc_per_node albalat/scripts/embed_chunks.py albalat/data/processed/embeddings albalat/data/processed/chunks.parquet

