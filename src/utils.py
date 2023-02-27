from typing import Dict, List, Union
import torch
from accelerate.utils.operations import _gpu_gather


def get_params_groups(
    model: torch.nn.Module, wd: float
) -> List[Dict[str, Union[List[torch.nn.Parameter], float]]]:
    """Splits model's parameters into separate groups where no weight decay is applied.

    Args:
        model (torch.nn.Module): Model.
        wd (float): Weight decay parameter.

    Returns:
        List[Dict[str, Union[List[torch.nn.Parameter], float]]]: List of dicts containing parameters and weigh decay coef.
    """
    # Adapted from https://github.com/facebookresearch/dino/blob/main/utils.py
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized, "weight_decay": wd},
        {"params": not_regularized, "weight_decay": 0.0},
    ]


class TempCoef(torch.nn.Module):
    """Module wrapping a temperature coeficient used to compute contrastive losses."""

    def __init__(self, initial_value: float) -> None:
        """Constructs TempCoef instance.

        Args:
            initial_value (float): Startting value of the temperature.
        """
        super().__init__()
        self.temp_coef = torch.nn.Parameter(torch.Tensor([initial_value]))

    def forward(self, logits_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module: Divide input tensor by the temperature value.

        Args:
            logits_matrix (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: _description_
        """
        # Apply learnable temperature factor on similarities
        # Clamping after op to avoid numerical instabilities
        logits_matrix = logits_matrix * self.temp_coef.clamp(1e-4, 30.0)

        return logits_matrix

    def get_temp_coef(self) -> float:
        """Get temperature value.

        Returns:
            float: temperature value.
        """
        return self.temp_coef.data.item()


def clip_contrastive_loss(
    emb_1: torch.Tensor, emb_2: torch.Tensor, temperature_coef: TempCoef
) -> torch.Tensor:
    """Computes contrastive CLIP-style loss.

    Args:
        emb_1 (torch.Tensor): Input embeddings.
        emb_2 (torch.Tensor): Embedding of positive pairs (perturbed inputs)

    Returns:
        torch.Tensor: Contrastive loss.
    """

    # Gathers embeddings across devices.
    emb_1_dist, emb_2_dist = _gpu_gather(
        (
            emb_1,
            emb_2,
        )
    )

    # Compute cosine similarity matrix
    similarities = emb_1_dist @ emb_2_dist.T

    similarities = temperature_coef(similarities)

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


def pool_and_normalize(
    features_sequence: torch.Tensor, attention_masks: torch.Tensor
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


def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu


def retrieval_eval(
    x_source: torch.Tensor, x_target: torch.Tensor
) -> List[torch.Tensor]:
    """Performs retrieval evaluation given paired embeddings of source and target data.

    Args:
        x_source (torch.Tensor): Source batch of embeddings with shape [B, emb_dim].
        x_target (torch.Tensor): Target batch of embeddings with shape [B, emb_dim].

    Returns:
        List[torch.Tensor]: Various retrieval metrics: R@1, R@5, and MRR.
    """

    # Compute similarity matrix
    similarities = x_source @ x_target.T

    topk_indices = torch.topk(similarities, k=similarities.size(1), dim=1)[1]

    ce_labels = torch.arange(similarities.size(0)).long().view(similarities.size(0), 1)

    # Bool tensor indicating which rows contain the idx corresponding to the main diag. of the sim matrix
    results = topk_indices.eq(ce_labels)

    r_at_1 = results[:, :1].sum() / float(similarities.size(0))
    r_at_5 = results[:, :5].sum() / float(similarities.size(0))

    ranks = results.nonzero()[:, 1].float() + 1.0
    mrr = (1 / ranks).mean()

    return r_at_1, r_at_5, mrr


def modify_optimizer_state_dict(state):

    modified_state = {"param_groups": state["param_groups"], "state": {}}

    for el in state["state"]:
        state_component = {}
        for k, v in state["state"][el].items():
            if isinstance(v, torch.Tensor):
                state_component[k] = v.cpu()
            else:
                state_component[k] = v

        modified_state["state"][el] = state_component

    return modified_state


def modify_model_state_dict(state):

    modified_state = state.__class__()
    for el in state:
        modified_state[el] = state[el].cpu()

    return modified_state
