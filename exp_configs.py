BASE_CONFIG = {
    "model_config": "bert-base-cased",
    "train_batch_size": 64,
    "test_batch_size": 64,
    "skip_steps": 1,  # Training steps to accumulate gradients through prior to updating params.
    "initial_temperature_coef": 1.0725,  # Matches initial value in clip.
    "use_projection": False,
    "alpha": 0.5,  # losses weigths in [0.0, 1.0]. Set to 1.0 to turn off contrastive loss. training_loss = alpha * mlm_loss + (1-alpha) * contrastive_loss
    "local_contrastive_loss": False,
    "learning_rate": 6e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 24_000,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-6,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,
    "fp16": False,
    "bf16": True,
    "n_workers": 2,
    "tokenizer_path": "bigcode/tokenizer-the-stack-march-sample",
    "mlm_masking_probability": 0.15,
    "contrastive_masking_probability": 0.3,
    "maximum_input_length": 384 * 2,
    "maximum_raw_length": 10_000,
}

MODEL_CONFIGS = {
    "mlm": {
        "alpha": 1.0,
        "mlm_masking_probability": 0.15,
    },
    "contrastive_local": {
        "alpha": 0.4,
        "initial_temperature_coef": 1.0725,  # Matches initial value in clip.
        "local_contrastive_loss": True,
        "mlm_masking_probability": 0.15,
        "contrastive_masking_probability": 0.2,
    },
    "contrastive_global": {
        "alpha": 0.4,
        "initial_temperature_coef": 1.0725,  # Matches initial value in clip.
        "local_contrastive_loss": False,
        "mlm_masking_probability": 0.15,
        "contrastive_masking_probability": 0.2,
    },
}

EXP_GROUPS = {}

for k, v in MODEL_CONFIGS.items():
    cfg = dict(BASE_CONFIG)
    cfg.update(v)
    EXP_GROUPS[k] = [cfg]
