import os
import torch
from torch.utils.data.dataset import Dataset
import transformers
from transformers import Trainer
from transformers import AutoConfig, BertForPreTraining
from typing import Tuple, Union, Dict, List

from transformers.trainer_utils import PredictionOutput
from transformers.modeling_utils import PreTrainedModel

from src.utils import (
    TempCoef,
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
        second indices respectively. Embedding norms are expected in index 2, and the value
        of the temperature coefficient in the final index.

    Returns:
        Dict[str, float]: Retrieval performance metrics.
    """

    recall_at_1, recall_at_5, mean_reciprocal_rank = retrieval_eval(
        torch.from_numpy(eval_pred.predictions[0]),
        torch.from_numpy(eval_pred.predictions[1]),
    )

    embedding_norms = eval_pred.predictions[2]

    temp_coef = eval_pred.predictions[-1]

    metrics = {
        "R@1": recall_at_1.item(),
        "R@5": recall_at_5.item(),
        "MRR": mean_reciprocal_rank.item(),
        "embedding_norms": [norm for norm in embedding_norms],
        "min_embedding_norm": embedding_norms.min().item(),
    }

    if temp_coef is not None:
        metrics["temp_coef"] = temp_coef.mean().item()

    return metrics


class CustomTrainer(Trainer):
    """Custom trainer class for training BERT with an additional contrastive loss."""

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: List[torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[float, torch.Tensor]]]:
        """Compute training loss. If return outputs is True, source/target embeddings are also returned.

        Args:
            model (PreTrainedModel): Training model
            inputs (List[torch.Tensor]): Inputs dict.
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

            normalized_source_target_embedding, embedding_norms = pool_and_normalize(
                source_target_embedding, source_target_att_mask, return_norms=True
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

            try:
                temp_coef = temp_coef_fn.get_temp_coef()
            except AttributeError:
                temp_coef = None

            return contrastive_loss, {
                "source_embedding": normalized_source_embedding,
                "target_embedding": normalized_target_embedding,
                "embedding_norms": embedding_norms,  # Used only for logging
                "temp_coef": temp_coef,  # Used only for logging
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

            if model.module.loss_alpha < 1.0:
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
            else:
                loss = out.loss

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
        pad_token_id=exp_dict["pad_token_id"],
        max_position_embeddings=exp_dict["maximum_input_length"],
        gradient_checkpointing=True,
        output_hidden_states=True,
    )

    encoder = BertForPreTraining(encoder_config)

    if exp_dict["alpha"] < 1.0:
        encoder.temperature_coef = TempCoef(
            initial_value=exp_dict["initial_temperature_coef"]
        )
    else:
        encoder.temperature_coef = torch.nn.Identity()

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
    max_steps: int,
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
        max_steps (int): Maximum number of training steps.
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
        max_steps=max_steps,
        learning_rate=exp_dict["learning_rate"],
        lr_scheduler_type=exp_dict["lr_scheduler_type"],
        warmup_steps=exp_dict["warmup_steps"],
        adam_beta1=exp_dict["adam_beta1"],
        adam_beta2=exp_dict["adam_beta2"],
        adam_epsilon=exp_dict["adam_epsilon"],
        weight_decay=exp_dict["weight_decay"],
        max_grad_norm=exp_dict["max_grad_norm"],
        gradient_accumulation_steps=exp_dict["skip_steps"],
        fp16=exp_dict["fp16"],
        bf16=exp_dict["bf16"],
        remove_unused_columns=False,
        label_names=[],
        deepspeed=deepspeed_cfg_path,
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=log_every,
        save_strategy="steps",
        save_steps=log_every,
        evaluation_strategy="steps",
        report_to="wandb",
    )

    encoder = get_encoder(exp_dict=exp_dict)

    trainer = CustomTrainer(
        model=encoder,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer
