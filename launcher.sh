#!/bin/bash

export WANDB_PROJECT=tf_encoder

PATH_TO_DATA=<Specify path>
PATH_TO_LOG=<Specify path>

TRAIN_DATA_NAME=bigcode/the-stack-march-sample
NGPUS=2
NSTEPS=100_000

torchrun --nproc_per_node $NGPUS \
trainval.py \
-e mlm \
--wandb-entity-name bigcode \
--wandb-project-name tf_encoder \
--wandb-run-name mlm \
--wandb-log-gradients false \
--steps $NSTEPS \
-sb $PATH_TO_LOG \
--train_data_name $TRAIN_DATA_NAME \
--data_path $PATH_TO_DATA
