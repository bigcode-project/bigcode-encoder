from typing import List, Dict
import tqdm
import logging
import math
import torch
from transformers import AutoConfig, BertForPreTraining
from accelerate import Accelerator
from src.utils import (
    get_params_groups,
    pooling,
    TempCoef,
    modify_model_state_dict,
    modify_optimizer_state_dict,
    retrieval_eval,
)
from torch.optim.lr_scheduler import *
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class BERT(torch.nn.Module):
    """Main model used for unsupervised learning of various types of representations.
    This class includes training methods.
    """

    def __init__(
        self,
        model_config: str,
        vocab_size: int,
        maximum_input_length: int,
        initial_temperature_coef: float,
        accelerator: Accelerator,
        alpha: float = 0.5,
    ) -> None:
        """Constructs model instance.

        Args:
            model_config (str): Model id.
            vocab_size (int): Number of tokens.
            maximum_input_length (int): Maximum length of input sequence of tokens.
            initial_temperature_coef (float): Initial value of the temperature. This will be trated as a learnable parameter.
            accelerator (Accelerator): Accelerator object used for data parallel.
            alpha (float, optional): Value in [0,1] weighing BERT's and contrastive losses. Defaults to 0.5.
        """
        super().__init__()

        encoder_config = AutoConfig.from_pretrained(
            model_config,
            vocab_size=vocab_size,
            max_position_embeddings=maximum_input_length,
            gradient_checkpointing=True,
            output_hidden_states=True,
        )
        self.encoder = BertForPreTraining(encoder_config)
        self.temperature_coef = TempCoef(initial_value=initial_temperature_coef)
        self.alpha = alpha
        self.accelerator = accelerator

    def train_on_batch(
        self,
        batch: List[torch.Tensor],
        skip_steps: int,
        update_parameters: bool,
        **extras,
    ) -> Dict[str, float]:
        """Trains model on batch if update_parameters==True, else accumulates gradients.

        Args:
            batch (List[torch.Tensor]): Batches of tokens, masks, and labels.
            skip_steps (int): How many steps to accumulate grads for before updating parameters.
            update_parameters (bool): Whether to update params or accumulate grads.

        Returns:
            Dict[str, float]: Losses dict.
        """

        self.train()

        (
            input_ids,
            att_masks,
            pooling_masks,
            positive_input_ids,
            positive_pooling_masks,
            mlm_labels,
            seq_relationship_labels,
        ) = batch

        with torch.no_grad():
            # Not taping gradients of positive pairs embedding op to avoid accelerator.backward's error
            # when running on more than one device.
            positive_embedding = self.encoder(
                input_ids=positive_input_ids, attention_mask=positive_pooling_masks
            ).hidden_states[-1]

            if self.use_projection:
                positive_embedding = self.projection_head(positive_embedding)

            normalized_positive_embedding = self.pool_and_normalize(
                positive_embedding,
                positive_pooling_masks,
            )

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=att_masks,
            labels=mlm_labels,
            next_sentence_label=seq_relationship_labels,
        )

        embedding = out.hidden_states[-1]
        if self.use_projection:
            embedding = self.projection_head(embedding)
        normalized_embedding = self.pool_and_normalize(
            embedding,
            pooling_masks,
        )

        contrastive_loss = self.clip_contrastive_loss(
            normalized_embedding,
            normalized_positive_embedding,
        )

        loss = out.loss * self.alpha + contrastive_loss * (1 - self.alpha)

        # Backward pass
        self.accelerator.backward(
            loss / skip_steps
        )  # Scales the grads for accumulation.

        if update_parameters:
            # Clip gradients
            self.accelerator.clip_grad_norm_(self.parameters(), self.grad_clip)

            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

        try:
            temp_coef = self.temperature_coef.module.get_temp_coef()
        except AttributeError:
            temp_coef = self.temperature_coef.get_temp_coef()

        return {
            "train_loss": loss.item(),
            "bert_loss": out.loss.item(),
            "cont_loss": contrastive_loss.item(),
            "temp_coef": temp_coef,
        }

    def train_on_loader(
        self,
        loader: torch.utils.data.DataLoader,
        skip_steps: int = 1,
        log_every: int = -1,
        **extras,
    ) -> Dict[str, float]:
        """Trains model for a full epoch on a given data loader.

        Args:
            loader (torch.utils.data.DataLoader): data loader.
            skip_steps (int, optional): Steps to wait before updating params. Defaults to 1.
            log_every (int, optional): Steps to wait before logging partial results. Defaults to -1.

        Returns:
            Dict[str, float]: Losses dictionary.
        """

        for iteration, batch in enumerate(
            tqdm.tqdm(
                loader, desc=f"Epoch {extras.get('epoch', 'unknown')}", leave=False
            )
        ):
            # Update params every skip_steps iters
            update_iteration = iteration % skip_steps == 0

            train_dict = self.train_on_batch(
                batch,
                skip_steps,
                update_iteration,
            )

            if log_every > 0 and iteration % log_every == 0:
                if self.accelerator.is_main_process:
                    logging.info(f"\nTraining scores: {train_dict}\n")

        train_dict.update({"lr": self.opt.param_groups[0]["lr"]})

        return train_dict

    @torch.no_grad()
    def eval_on_loader(self, loader, **extras):
        self.eval()

        contrastive_loss_list = []
        source_embeddings_list = []
        target_embeddings_list = []

        for batch in loader:
            (
                source_inputs,
                source_att_mask,
                target_inputs,
                target_att_mask,
            ) = batch

            source_embedding = self.encoder(
                input_ids=source_inputs, attention_mask=source_att_mask
            ).hidden_states[-1]
            target_embedding = self.encoder(
                input_ids=target_inputs, attention_mask=target_att_mask
            ).hidden_states[-1]

            if self.use_projection:
                source_embedding = self.projection_head(source_embedding)
                target_embedding = self.projection_head(target_embedding)

            normalized_source_embedding = self.pool_and_normalize(
                source_embedding,
                source_att_mask,
            )
            normalized_target_embedding = self.pool_and_normalize(
                target_embedding,
                target_att_mask,
            )

            contrastive_loss = self.clip_contrastive_loss(
                normalized_source_embedding,
                normalized_target_embedding,
            )

            (
                gathered_source_embedding,
                gathered_target_embedding,
            ) = self.accelerator.gather(
                (normalized_source_embedding, normalized_target_embedding)
            )

            source_embeddings_list.append(gathered_source_embedding.cpu())
            target_embeddings_list.append(gathered_target_embedding.cpu())

            contrastive_loss = contrastive_loss.repeat(source_inputs.size(0))
            contrastive_loss_list.append(self.accelerator.gather(contrastive_loss))

        source_embedding = torch.cat(source_embeddings_list, 0)
        target_embedding = torch.cat(target_embeddings_list, 0)

        recall_at_1, recall_at_5, mean_reciprocal_rank = retrieval_eval(
            source_embedding, target_embedding
        )

        logging_prefix = extras.get("logging_prefix", "test")

        return {
            f"{logging_prefix}_contrastive_loss": torch.mean(
                torch.cat(contrastive_loss_list)
            ).item(),
            f"{logging_prefix}_R@1": recall_at_1.item(),
            f"{logging_prefix}_R@5": recall_at_5.item(),
            f"{logging_prefix}_MRR": mean_reciprocal_rank.item(),
        }

    def forward(
        self, ids: torch.Tensor, att_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward data through the model.

        Args:
            ids (torch.Tensor): Input tokens.
            att_mask (torch.Tensor): Padding masks

        Returns:
            Dict[str, torch.Tensor]: Dict with various outputs from different parts of the model.
        """
        return self.encoder(input_ids=ids, attention_mask=att_mask)

    def clip_contrastive_loss(
        self, emb_1: torch.Tensor, emb_2: torch.Tensor
    ) -> torch.Tensor:
        """Computes contrastive CLIP-style loss.

        Args:
            emb_1 (torch.Tensor): Input embeddings.
            emb_2 (torch.Tensor): Embedding of positive pairs (perturbed inputs)

        Returns:
            torch.Tensor: Contrastive loss.
        """

        emb_1_dist, emb_2_dist = self.accelerator.gather(
            (
                emb_1,
                emb_2,
            )
        )

        # Compute cosine similarity matrix
        similarities = emb_1_dist @ emb_2_dist.T

        similarities = self.temperature_coef(similarities)

        # Matching representations of positive pairs assumed to be located at the main
        # dioagonal of the similarity matrix if targets are not given
        ce_labels = torch.arange(similarities.size(0)).long().to(similarities.device)

        # We use a cross-entropy criterion to increase the similarities between
        # matching representations of source and target
        sim_loss = 0.5 * (
            torch.nn.functional.cross_entropy(similarities, ce_labels)
            + torch.nn.functional.cross_entropy(similarities.T, ce_labels)
        )

        return sim_loss

    def info_nce_contrastive_loss(
        self, emb_1: torch.Tensor, emb_2: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Computes contrastive InfoNCE loss.

        Args:
            emb_1 (torch.Tensor): Input embeddings.
            emb_2 (torch.Tensor): Embedding of positive pairs (perturbed inputs)
            eps (float, optional): Used to avoid numerical issues. Defaults to 1e-6.

        Returns:
            torch.Tensor: Contrastive loss.
        """

        emb_1_dist, emb_2_dist = self.accelerator.gather(
            (
                emb_1,
                emb_2,
            )
        )

        emb = torch.cat([emb_1, emb_2], dim=0)
        out_dist = torch.cat([emb_1_dist, emb_2_dist], dim=0)

        cov = torch.mm(emb, out_dist.t().contiguous())
        cov = self.temperature_coef(cov)
        sim = torch.exp(cov)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.ones_like(neg) * (math.e ** (self.temperature_coef(1.0)))
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(self.temperature_coef(torch.sum(emb_1 * emb_2, dim=-1)))
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def pool_and_normalize(
        self, features_sequence: torch.Tensor, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        """Temporal ooling of sequences of vectors and projection onto the unit sphere.

        Args:
            features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
            attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].

        Returns:
            torch.Tensor: Pooled and normalized vectors with shape [B, F].
        """

        pooled_embeddings = pooling(features_sequence, attention_masks)
        pooled_normalized_embeddings = (
            pooled_embeddings / pooled_embeddings.norm(dim=1)[:, None]
        )

        return pooled_normalized_embeddings

    def get_state_dict(self):

        # Puts state tensors in the cpu to avoid gpu:0 OOM when loading back.
        unwrapped_model_state = modify_model_state_dict(
            self.accelerator.unwrap_model(self.encoder).state_dict()
        )
        unwrapped_projection_head_state = modify_model_state_dict(
            self.accelerator.unwrap_model(self.projection_head).state_dict()
        )
        unwrapped_temp_coef_state = modify_model_state_dict(
            self.accelerator.unwrap_model(self.temperature_coef).state_dict()
        )
        optimizer_state = modify_optimizer_state_dict(self.opt.state_dict())

        state_dict = {
            "projection_head": unwrapped_projection_head_state,
            "temperature_coef": unwrapped_temp_coef_state,
            "model": unwrapped_model_state,
            "opt": optimizer_state,
            "scheduler": self.scheduler.state_dict(),
        }

        return state_dict

    def set_state_dict(self, state_dict):
        self.projection_head.load_state_dict(state_dict["projection_head"])
        self.encoder.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.temperature_coef.load_state_dict(state_dict["temperature_coef"])


def get_model(exp_dict: dict, accelerator: Accelerator) -> BERT:
    """Intanstiates model object.

    Args:
        exp_dict (dict): Config dictionary.
        accelerator (Accelerator): Accelerator object.

    Returns:
        BERT: Model instance.
    """
    model = BERT(
        exp_dict["model_config"],
        exp_dict["vocab_size"],
        exp_dict["maximum_input_length"],
        exp_dict["initial_temperature_coef"],
        accelerator,
        alpha=exp_dict["alpha"],
    )

    feature_dim = model.encoder.config.hidden_size

    model.projection_head = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, feature_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_dim, feature_dim),
    )

    model.opt = torch.optim.AdamW(
        get_params_groups(
            model,
            exp_dict["l2"],
        ),
        lr=exp_dict["base_lr"],
        betas=exp_dict["betas"],
        amsgrad=exp_dict["amsgrad"],
    )

    model.scheduler = globals()[exp_dict["scheduler_config"]["name"]](
        model.opt, **exp_dict["scheduler_config"]["kwargs"]
    )

    model.use_projection = exp_dict["use_projection"]
    model.grad_clip = exp_dict["grad_clip"]

    return model
