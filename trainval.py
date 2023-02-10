import os
import logging
import argparse
import tqdm
import torch
import exp_configs
from accelerate import Accelerator
from src import datasets_loader, models
from src.constants import RESULTS_FNAME
from haven import haven_wizard as hw


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    accelerator = Accelerator()

    logging.info(f"Accelerator info: {accelerator.device}, {accelerator.num_processes}")

    # Create data loader and model
    train_data = datasets_loader.get_dataset(
        path_to_cache=args.data_path,
        split="train",
        maximum_raw_length=exp_dict["maximum_raw_length"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=exp_dict["train_batch_size"],
        num_workers=exp_dict["n_workers"],
        collate_fn=datasets_loader.Collator(
            tokenizer_path=exp_dict["tokenizer_path"],
            maximum_length=exp_dict["maximum_length"],
            mlm_masking_probability=exp_dict["mlm_masking_probability"],
            contrastive_masking_probability=exp_dict["contrastive_masking_probability"],
        ),
        drop_last=True,
    )

    # Configures scheduler with the correct number of iterations rather than epochs.
    iterations_per_epoch = len(train_loader) // (
        accelerator.num_processes * exp_dict["skip_steps"]
    )
    if "max_epochs" in exp_dict["scheduler_config"]["kwargs"]:
        exp_dict["scheduler_config"]["kwargs"]["max_epochs"] = (
            args.epochs * iterations_per_epoch
        )
    if "warmup_epochs" in exp_dict["scheduler_config"]["kwargs"]:
        exp_dict["scheduler_config"]["kwargs"]["warmup_epochs"] = (
            exp_dict["scheduler_config"]["kwargs"]["warmup_epochs"]
            * iterations_per_epoch
        )

    exp_dict["vocab_size"] = len(train_loader.collate_fn.tokenizer.vocab)

    model = models.get_model(
        exp_dict=exp_dict,
        accelerator=accelerator,
    )

    # Resume or initialize checkpoint
    cm = hw.CheckpointManager(savedir)
    state_dict = cm.load_model()
    if state_dict is not None:
        model.set_state_dict(state_dict)

    (
        model.encoder,
        model.similarities_coef,
        model.projection_head,
        model.opt,
        train_loader,
    ) = model.accelerator.prepare(
        model.encoder,
        model.similarities_coef,
        model.projection_head,
        model.opt,
        train_loader,
    )

    # Train and Validate
    for epoch in tqdm.tqdm(
        range(cm.get_epoch(), args.epochs), desc="Running Experiment"
    ):
        # Train for one epoch
        train_dict = model.train_on_loader(
            train_loader,
            epoch=epoch,
            skip_steps=exp_dict["skip_steps"],
            log_every=args.log_every,
        )

        if model.accelerator.is_main_process:

            # Get Metrics
            score_dict = {
                "epoch": epoch,
            }
            score_dict.update(train_dict)

            # Save Metrics in "savedir" as score_list.pkl
            cm.log_metrics(score_dict)

            model.accelerator.save(
                model.get_state_dict(), os.path.join(savedir, "model.pth")
            )

    print("Experiment done\n")


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
