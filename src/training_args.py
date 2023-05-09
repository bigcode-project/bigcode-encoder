import argparse


def parse_bool_flag(s: str) -> bool:
    """Parse boolean arguments from the command line.

    Args:
        s (str): Input arg string.

    Returns:
        bool: _description_
    """
    _FALSY_STRINGS = {"off", "false", "0"}
    _TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in _FALSY_STRINGS:
        return False
    elif s.lower() in _TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def parse_args():
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
        "--train_data_name",
        type=str,
        default="bigcode/the-stack-march-sample",
        help="Name of training dataset.",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default=None,
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
        "--steps", default=500_000, type=int, help="Number of training steps."
    )
    parser.add_argument(
        "--log_every",
        default=1000,
        type=int,
        help="Number of iterations to wait before logging training scores.",
    )
    parser.add_argument(
        "--wandb-entity-name",
        type=str,
        default=None,
        help="Name of wandb entity for reporting.",
    )
    parser.add_argument(
        "--wandb-project-name", type=str, default=None, help="Name of wandb project."
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Name of run.")
    parser.add_argument(
        "--wandb-log-gradients",
        type=parse_bool_flag,
        default="false",
        help="Whether to write gradients to wandb logs.",
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

    return parser.parse_known_args()
