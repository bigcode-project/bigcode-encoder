import os
import torch
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import *
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import transformers
from transformers import Trainer
from transformers import AutoConfig, BertForPreTraining
from typing import Tuple, Union, Dict

from transformers.trainer_utils import PredictionOutput
from transformers.modeling_utils import PreTrainedModel

from src.utils import (
    TempCoef,
    get_params_groups,
    pool_and_normalize,
    clip_contrastive_loss,
    retrieval_eval,
)
from src.datasets_loader import Collator


def compute_metrics(eval_pred: PredictionOutput) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        eval_pred (PredictionOutput): Outputs from Trainer.predict(). The field
        'predictions' should contain source and target embeddings in the first and
        second indices respectively.

    Returns:
        Dict[str, float]: Retrieval performance metrics.
    """

    recall_at_1, recall_at_5, mean_reciprocal_rank = retrieval_eval(
        torch.from_numpy(eval_pred.predictions[0]),
        torch.from_numpy(eval_pred.predictions[1]),
    )

    return {
        "R@1": recall_at_1.item(),
        "R@5": recall_at_5.item(),
        "MRR": mean_reciprocal_rank.item(),
    }


class CustomTrainer(Trainer):
    """Custom trainer class for training BERT with an additional contrastive loss."""

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[float, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[float, torch.Tensor]]]:
        """Compute training loss. If return outputs is True, source/target embeddings are also returned.

        Args:
            model (PreTrainedModel): Training model
            inputs (Dict[float, torch.Tensor]): Inputs dict.
            return_outputs (bool, optional): Whether to return outputs for evaluation. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[float, torch.Tensor]]]: Training loss and optionally embeddings.
        """

        try:
            projection_fn = model.module.projection_head
            temp_coef_fn = model.module.temperature_coef
        except AttributeError:
            projection_fn = model.projection_head
            temp_coef_fn = model.temperature_coef

        if return_outputs:  # This branch is called during evaluation.
            source_target_ids, source_target_att_mask = inputs

            source_target_embedding = model(
                input_ids=source_target_ids, attention_mask=source_target_att_mask
            ).hidden_states[-1]

            source_target_embedding = projection_fn(source_target_embedding)

            normalized_source_target_embedding = pool_and_normalize(
                source_target_embedding,
                source_target_att_mask,
            )

            # Batches are such that the first and second halves are independently perturbed versions
            # of the same source data and are treated as positive pairs.
            pair_split_idx = source_target_embedding.size(0) // 2
            normalized_source_embedding = normalized_source_target_embedding[
                :pair_split_idx
            ]
            normalized_target_embedding = normalized_source_target_embedding[
                pair_split_idx:
            ]

            contrastive_loss = clip_contrastive_loss(
                normalized_source_embedding,
                normalized_target_embedding,
                temp_coef_fn,
                model.local_contrastive_loss,
            )

            return contrastive_loss, {
                "source_embedding": normalized_source_embedding,
                "target_embedding": normalized_target_embedding,
            }

        else:  # Training branch.
            (
                input_ids,
                attention_mask,
                pooling_mask,
                labels,
                next_sentence_label,
            ) = inputs

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                next_sentence_label=next_sentence_label,
            )
            embedding = out.hidden_states[-1]
            embedding = model.module.projection_head(embedding)
            normalized_embedding = pool_and_normalize(
                embedding,
                pooling_mask,
            )

            # Batches are such that the first and second halves are independently perturbed versions
            # of the same source data and are treated as positive pairs.
            pair_split_idx = embedding.size(0) // 2
            contrastive_loss = clip_contrastive_loss(
                normalized_embedding[:pair_split_idx],
                normalized_embedding[pair_split_idx:],
                model.module.temperature_coef,
                model.module.local_contrastive_loss,
            )

            loss = out.loss * model.module.loss_alpha + contrastive_loss * (
                1 - model.module.loss_alpha
            )

            return loss


def get_encoder(exp_dict: dict) -> PreTrainedModel:

    """get encoder given config exp_dict.

    Args:
        exp_dict (dict): Exp config. Those are set in the module exp_configs.py

    Returns:
        PreTrainedModel: Model to be trained.
    """

    encoder_config = AutoConfig.from_pretrained(
        exp_dict["model_config"],
        vocab_size=exp_dict["vocab_size"],
        max_position_embeddings=exp_dict["maximum_input_length"],
        gradient_checkpointing=True,
        output_hidden_states=True,
    )

    encoder = BertForPreTraining(encoder_config)

    encoder.temperature_coef = TempCoef(
        initial_value=exp_dict["initial_temperature_coef"]
    )

    if exp_dict["use_projection"]:
        feature_dim = encoder.config.hidden_size
        encoder.projection_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_dim, feature_dim),
        )
    else:
        encoder.projection_head = torch.nn.Identity()

    encoder.loss_alpha = exp_dict["alpha"]

    encoder.local_contrastive_loss = exp_dict["local_contrastive_loss"]

    return encoder


def get_trainer(
    exp_dict: dict,
    savedir: str,
    epochs: int,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    collate_fn: Collator,
    log_every: int = 100,
    local_rank: int = 0,
    deepspeed_cfg_path: str = None,
) -> CustomTrainer:
    """Intanstiates Trainer object.

    Args:
        exp_dict (dict): Config dictionary.
        savedir (str): Output path.
        epochs (int): Maximum number of training epochs.
        train_dataset (Dataset): Training data.
        valid_dataset (Dataset): Evaluation data.
        collate_fn (Collator): Collator.
        log_every (int): Logging interval.
        local_rank (int): Device id for distributed training.
        deepspeed_cfg_path (str, Optional): Optional path to deepspeed config.

    Returns:
        CustomTrainer: Trainer object.
    """

    training_args = transformers.TrainingArguments(
        output_dir=savedir,
        local_rank=local_rank,
        per_device_train_batch_size=exp_dict["train_batch_size"],
        per_device_eval_batch_size=exp_dict["test_batch_size"],
        num_train_epochs=epochs,
        gradient_accumulation_steps=exp_dict["skip_steps"],
        max_grad_norm=exp_dict["grad_clip"],
        remove_unused_columns=False,
        label_names=[],
        fp16=True,
        deepspeed=deepspeed_cfg_path,
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=log_every,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    encoder = get_encoder(exp_dict=exp_dict)

    opt = torch.optim.AdamW(
        get_params_groups(
            encoder,
            exp_dict["l2"],
        ),
        lr=exp_dict["base_lr"],
        betas=exp_dict["betas"],
        amsgrad=exp_dict["amsgrad"],
    )

    lr_scheduler = globals()[exp_dict["scheduler_config"]["name"]](
        opt, **exp_dict["scheduler_config"]["kwargs"]
    )

    trainer = CustomTrainer(
        model=encoder,
        args=training_args,
        optimizers=(opt, lr_scheduler),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer
