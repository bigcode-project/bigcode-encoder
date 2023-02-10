source activate /mnt/home/envs/accelerate

python trainval.py \
-e base \
--epochs 100 \
-sb /mnt/colab_public/results/joao/bigcode_bert \
-r 1 \
--data_path /mnt/colab_public/datasets/joao/CodeSearchNet \
--python_binary 'source activate /mnt/home/envs/accelerate ; /mnt/home/envs/accelerate/bin/accelerate launch --config_file ./resources/accelerate_configs/accelerate_config.yaml' \
--j toolkit
