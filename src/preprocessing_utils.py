from typing import Dict, List, Union
from matplotlib import docstring
import torch
from src.constants import PADDING_ID_FOR_LABELS


def perturb_tokens(
    input_ids: torch.Tensor,
    att_masks: torch.Tensor,
    masking_fraction: float,
    masking_token_id: Union[int, float],
) -> List[torch.Tensor]:
    """Perturb tokens in preparation for MLM loss computation.

    Args:
        input_ids (torch.Tensor): Batch of input tokens IDs.
        att_masks (torch.Tensor): Input padding masks.
        masking_fraction (float): Probability of masking ou a given token.
        masking_token_id (Union[int, float]): Token id for masks.

    Returns:
        List[torch.Tensor]: Perturbed ids along with label ids.
    """
    modified_att_masks = att_masks.clone()
    modified_att_masks[:, 0] = 0  # Don't mask the first token given by [CLS].
    modified_att_masks[
        torch.arange(att_masks.size(0), device=att_masks.device), att_masks.sum(1) - 1
    ] = 0  # Don't mask the last token given by [SEP].

    element_wise_bernoulli = torch.distributions.Bernoulli(
        masking_fraction * modified_att_masks
    )  # Independent Bernoulli random variables for each token.

    mask_ids = torch.ones_like(input_ids) * masking_token_id
    padding_ids = torch.ones_like(input_ids) * PADDING_ID_FOR_LABELS
    perturbation_idx = element_wise_bernoulli.sample().bool()  # Sample random mask.
    mlm_labels = torch.where(perturbation_idx, input_ids.clone(), mask_ids)
    mlm_labels = torch.where(
        att_masks == 0, padding_ids, mlm_labels
    )  # Places padding labels.
    perturbed_ids = torch.where(perturbation_idx, mask_ids, input_ids.clone())

    return perturbed_ids, mlm_labels


def perturb_words(
    input_sentence: str, masking_fraction: float, masking_token: Union[int, float]
) -> str:
    """Perturbs words in a given sentence.

    Args:
        input_sentence (str): input sentence.
        masking_fraction (float): Probability of masking out any given word.
        masking_token (Union[int, float]): Token used for masking. E.g., '<mask>' or [MASK].

    Returns:
        str: Perturbed version of input sentence.
    """

    output_words = []

    for word in input_sentence.split(" "):
        if torch.rand(1).item() < masking_fraction:
            output_words.append(masking_token)
        else:
            output_words.append(word)

    return " ".join(output_words)


def truncate_sentences(
    sentence_list: List[str], maximum_length: Union[int, float]
) -> List[str]:
    """Truncates list of sentences to a maximum length.

    Args:
        sentence_list (List[str]): List of sentences to be truncated.
        maximum_length (Union[int, float]): Maximum length of any output sentence.

    Returns:
        List[str]: List of truncated sentences.
    """

    truncated_sentences = []

    for sentence in sentence_list:
        truncated_sentences.append(sentence[:maximum_length])

    return truncated_sentences


def split_sentence(
    sentence: str, maximum_length: Union[int, float] = None
) -> List[str]:
    """Truncates and splits a given sentence.

    Args:
        sentence (str): Input sentence.
        maximum_length (Union[int, float], optional): Maximum length. Defaults to None.

    Returns:
        List[str]: List of pair of sentences, each being a half of the input after truncation.
    """

    if maximum_length is None:
        maximum_length = len(sentence)
    else:
        maximum_length = min(maximum_length, len(sentence))

    half_length = maximum_length // 2

    return sentence[:half_length], sentence[half_length:maximum_length]


def get_pooling_mask(
    input_ids: torch.Tensor, sep_token_id: Union[int, float]
) -> torch.Tensor:
    """Gets pooling masks. For a sequence of input tokens, the mask will be
    a sequence of ones up until the first [SEP] occurrence, and 0 after that.

    Args:
        input_ids (torch.Tensor): Batch of input ids with shape [B, T].
        sep_token_id (Union[int, float]): Id for [SEP] token.

    Returns:
        torch.Tensor: Batch of pooling masks with shape [B, T]
    """
    # idx indicates the first occurrence of sep_token_id per along dim 0 of input_ids
    idx = (input_ids == sep_token_id).float().argmax(1)

    repeated_idx = idx.unsqueeze(1).repeat(1, input_ids.size(1))

    ranges = torch.arange(input_ids.size(1)).repeat(input_ids.size(0), 1)

    pooling_mask = (repeated_idx >= ranges).long()

    return pooling_mask


class pre_process_codesearchnet_train:
    def __init__(self, maximum_length: int) -> None:
        """Pre process code search net data by truncating and splitting code snippets.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads code string, truncates it and splits in two pieces.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """
        code_str = example["func_code_string"]
        code_str_source, code_str_target = split_sentence(code_str, self.maximum_length)
        example.update({"source": code_str_source, "target": code_str_target})
        return example


class pre_process_codesearchnet_test:
    def __init__(self, maximum_length: int) -> None:
        """Pre process code search net data by truncating and pairing code and docstring.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads and truncates code and doc strings.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """
        source = example["func_documentation_tokens"]
        source = (" ").join(source)[: self.maximum_length]
        target = example["func_code_string"][: self.maximum_length]
        example.update({"source": source, "target": target})
        return example


class pre_process_gfg:
    def __init__(self, maximum_length: int) -> None:
        """Pre process Python-Java Geeks4Geeks data by truncating and pairing code snippets.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads and truncates code strings.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """

        source = example["python_func"][: self.maximum_length]
        target = example["java_func"][: self.maximum_length]

        example.update({"source": source, "target": target})

        return example
