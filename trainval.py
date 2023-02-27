import sys
import os
import logging
import argparse
import exp_configs
from src import datasets_loader, hf_trainer
from src.constants import RESULTS_FNAME, GFG_DATA_PATH, MAX_VALID_DATA_ROW_COUNT
from haven import haven_wizard as hw

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    # Create data loaders and model
    train_data = datasets_loader.get_dataset(
        dataset_name="code_search_net",
        path_to_cache=args.data_path,
        split="train",
        maximum_raw_length=exp_dict["maximum_raw_length"],
    )
    gfg_test_data = datasets_loader.get_dataset(  # Geeks4Geeks data
        dataset_name="gfg",
        path_to_cache=GFG_DATA_PATH,
        split="test",
        maximum_raw_length=exp_dict["maximum_raw_length"],
        maximum_row_cout=MAX_VALID_DATA_ROW_COUNT,
    )
    collate_fn = datasets_loader.Collator(
        tokenizer_path=exp_dict["tokenizer_path"],
        maximum_length=exp_dict["maximum_input_length"],
    )

    exp_dict["vocab_size"] = collate_fn.vocabulary_size

    trainer = hf_trainer.get_trainer(
        exp_dict=exp_dict,
        savedir=savedir,
        epochs=args.epochs,
        train_dataset=train_data,
        valid_dataset=gfg_test_data,
        collate_fn=collate_fn,
        log_every=args.log_every,
        local_rank=args.local_rank,
        deepspeed_cfg_path=args.deepspeed,
    )

    trainer.train(
        resume_from_checkpoint=any(
            dir.startswith("checkpoint") for dir in os.listdir(savedir)
        )
    )

    logging.info("Experiment done\n")


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

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "toolkit":
        import job_configs

        job_config = job_configs.JOB_CONFIG[args.exp_group]

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname=RESULTS_FNAME,
        python_binary_path=args.python_binary,
        args=args,
    )
