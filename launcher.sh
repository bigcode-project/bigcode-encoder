#!/bin/bash

PATH_TO_DATA=<Specify path>
PATH_TO_LOG=<Specify path>

torchrun --nproc_per_node 2 \
trainval.py \
-e base \
--epochs 100 \
-sb $PATH_TO_LOG \
--data_path $PATH_TO_DATA
