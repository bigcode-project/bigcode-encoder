BASE_CONFIG = {
    "model_config": "bert-base-cased",
    "train_batch_size": 64,
    "test_batch_size": 64,
    "skip_steps": 3,  # Training steps to accumulate gradients through prior to updating params.
    "initial_temperature_coef": 10.0,
    "use_projection": False,
    "alpha": 0.5,  # losses weigths in [0.0, 1.0]. Set to 1.0 to turn off contrastive loss. training_loss = alpha * mlm_loss + (1-alpha) * contrastive_loss
    "local_contrastive_loss": False,
    "learning_rate": 6e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 24_000,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-08,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,
    "n_workers": 2,
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
