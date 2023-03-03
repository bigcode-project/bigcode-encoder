import os
import argparse
import exp_configs

from trainval_toolkit import train


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        help="Define the base directory where data will be cached.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Choose Job Scheduler."
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--log_every",
        default=1000,
        type=int,
        help="Number of iterations to wait before logging training scores.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--deepspeed",
        default=None,
        type=str,
        help="""Optional path to deepspeed config.""",
    )

    args, others = parser.parse_known_args()

    train(
        exp_configs.EXP_GROUPS[args.exp_group][0],
        os.path.join(args.savedir_base),
        args,
    )
