#!/bin/bash

PATH_TO_DATA=<Specify path>
PATH_TO_LOG=<Specify path>

TRAIN_DATA_NAME=bigcode/the-stack-march-sample
NGPUS=2
NEPOCHS=100

torchrun --nproc_per_node $NGPUS \
trainval.py \
-e base \
--epochs $NEPOCHS \
-sb $PATH_TO_LOG \
--train_data_name $TRAIN_DATA_NAME \
--data_path $PATH_TO_DATA
