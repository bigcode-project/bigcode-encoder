import sys
import os
import logging
import argparse
import exp_configs
from src import datasets_loader, hf_trainer
from src.training_args import parse_args
from src.constants import RESULTS_FNAME, GFG_DATA_PATH, MAX_VALID_DATA_ROW_COUNT
from haven import haven_wizard as hw

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    # Create data loaders and model
    train_data = datasets_loader.get_dataset(
        dataset_name=args.train_data_name,
        path_to_cache=None,
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
        mlm_masking_probability=exp_dict["mlm_masking_probability"],
        contrastive_masking_probability=exp_dict["contrastive_masking_probability"],
        ignore_contrastive_loss_data=exp_dict["alpha"] == 1.0,
    )

    exp_dict["vocab_size"] = collate_fn.vocabulary_size
    exp_dict["pad_token_id"] = collate_fn.pad_token_id

    trainer = hf_trainer.get_trainer(
        exp_dict=exp_dict,
        savedir=savedir,
        max_steps=args.steps,
        train_dataset=train_data,
        valid_dataset=gfg_test_data,
        collate_fn=collate_fn,
        log_every=args.log_every,
        wandb_entity_name=args.wandb_entity_name,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=args.wandb_run_name,
        wandb_log_grads=args.wandb_log_gradients,
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
    args, others = parse_args()

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        args.local_rank = 0

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "toolkit":
        import job_configs

        job_config = job_configs.JOB_CONFIG[args.exp_group]

    # Run experiments and create results file
    hw.run_wizard(
        func=train,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname=RESULTS_FNAME,
        python_binary_path=args.python_binary,
        args=args,
    )
