source activate /mnt/home/envs/accelerate

python trainval.py \
-e base \
--epochs 100 \
-sb /mnt/home/exp_data/bigcode_bert \
-r 1 \
--data_path /mnt/colab_public/datasets/joao/CodeSearchNet \
--python_binary 'source activate /mnt/home/envs/accelerate ; /mnt/home/envs/accelerate/bin/python -m torch.distributed.launch --nproc_per_node 2' \
--j toolkit
