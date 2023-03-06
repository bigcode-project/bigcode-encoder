BASE_CONFIG = {
    "model_config": "bert-base-cased",
    "train_batch_size": 48,
    "test_batch_size": 48,
    "skip_steps": 3,  # Training steps to accumulate gradients through prior to updating params.
    "initial_temperature_coef": 10.0,
    "use_projection": False,
    "alpha": 0.5,
    "local_contrastive_loss": False,
    "base_lr": 5e-5,
    "scheduler_config": {
        "name": "LinearWarmupCosineAnnealingLR",
        "kwargs": {
            "warmup_epochs": 5,  # Will be converted tothe equivalent number of iterations.
            "max_epochs": 0,  # This will be set prior to training since it requires data information.
            "warmup_start_lr": 0.0,
            "eta_min": 0.0,
            "last_epoch": -1,
        },
    },
    "betas": [0.9, 0.999],
    "l2": 5e-3,
    "amsgrad": False,
    "n_workers": 2,
    "grad_clip": 1.0,
    "tokenizer_path": "bigcode/tokenizer-the-stack-march-sample",
    "mlm_masking_probability": 0.15,
    "contrastive_masking_probability": 0.3,
    "maximum_input_length": 384 * 2,
    "maximum_raw_length": 10000,
}

EXP_GROUPS = {
    "base": [],
}

EXP_GROUPS["base"].append(BASE_CONFIG)
