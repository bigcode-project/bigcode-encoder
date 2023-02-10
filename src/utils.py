from typing import Dict, List, Union
import torch


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


class TempCoef(torch.nn.Module):
    """Module wrapping a temperature coeficient used to compute the InfoNCE during training."""

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
        logits_matrix /= self.temp_coef.clamp(1e-4, 1e2)

        return logits_matrix

    def get_temp_coef(self) -> float:
        """Get temperature value.

        Returns:
            float: temperature value.
        """
        return self.temp_coef.data.item()


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
