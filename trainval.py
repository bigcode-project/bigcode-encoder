import os
import argparse
import exp_configs

from trainval_toolkit import train
from src.training_args import parse_args


if __name__ == "__main__":

    args, others = parse_args()

    train(
        exp_configs.EXP_GROUPS[args.exp_group][0],
        os.path.join(args.savedir_base),
        args,
    )
