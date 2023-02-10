from typing import List, Dict
import torch
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from src.preprocessing_utils import (
    perturb_words,
    perturb_tokens,
    get_pooling_mask,
    pre_process_codesearchnet,
)
from src.constants import MASK_TOKEN, PAD_TOKEN, SEPARATOR_TOKEN, CLS_TOKEN

DATASET_NAME_TO_PREPROCESSING_FUNCTION = {
    "code_search_net": pre_process_codesearchnet,
}


class dataset(Dataset):
    """Indexed dataset class."""

    def __init__(self, base_dataset: datasets.Dataset) -> None:
        """Intanstiates an indexed dataset wrapping a base data source.
        We use this class to be able to get examples from the dataset including negative pairs.

        Args:
            base_dataset (datasets.Dataset): Base indexed data source.
        """
        self.data_source = base_dataset

    def __len__(self) -> int:
        """Returns the length of the dataset which matches that of the base data source.

        Returns:
            int: Dataset length.
        """
        return len(self.data_source)

    def __getitem__(self, i: int) -> List[Dict]:
        """Reads from the base dataset and returns an addition random entry that serves as negative example.

        Args:
            i (int): Index to be read.

        Returns:
            List[Dict]: Pair of examples. The example indexed by i is returned along with a different random point.
        """
        rand_idx = torch.randint(0, len(self.data_source), (1,)).item()
        while rand_idx == i:
            rand_idx = torch.randint(0, len(self.data_source), (1,)).item()

        example = self.data_source[i]
        negative_example = self.data_source[rand_idx]

        return example, negative_example


def get_dataset(
    path_to_cache: str,
    split: str,
    maximum_raw_length: int,
    force_preprocess: bool = False,
) -> dataset:
    """Get dataset instance.

    Args:
        path_to_cache (str): Path to the base dataset.
        split (str): data split in {'train', 'valid', 'test'}.
        maximum_raw_length (int, optional): Maximum length of the raw entries from the source dataset.
        force_preprocess (bool, optional): Whether to force pre-processing. Defaults to False.

    Returns:
        dataset: An indexed dataset object.
    """
    codesearchnet_dataset = load_dataset("code_search_net", cache_dir=path_to_cache)

    if force_preprocess:
        codesearchnet_dataset.cleanup_cache_files()

    codesearchnet_dataset[split] = codesearchnet_dataset[split].map(
        DATASET_NAME_TO_PREPROCESSING_FUNCTION["code_search_net"](maximum_raw_length),
    )

    return dataset(
        codesearchnet_dataset[split],
    )


class Collator:
    """Collator object mapping sequences of items from dataset instance
    into batches of IDs and masks used for training models.
    """

    def __init__(
        self,
        tokenizer_path: str,
        maximum_length: int,
        mlm_masking_probability: float,
        contrastive_masking_probability: float,
    ) -> None:
        """Creates instance of collator.

        Args:
            tokenizer_path (str): Path to tokenizer.
            maximum_length (int): Truncating length of token sequences.
            mlm_masking_probability (float): Masking probability for MLM objective.
            contrastive_masking_probability (float): Masking probability for contrastive objective.
        """
        self.mlm_masking_probability = mlm_masking_probability
        self.contrastive_masking_probability = contrastive_masking_probability
        self.maximum_length = maximum_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
        self.tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
        self.tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})

        self.sep_token_id = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        self.mask_token_id = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.cla_token_id = self.tokenizer.get_vocab()[self.tokenizer.cls_token]

    def __call__(self, batch: List[Dict]) -> List[torch.Tensor]:
        """Maps list of pairs of examples to batches of token ids, masks, and labels used for training.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            List[torch.Tensor]: Batches of tokens, masks, and labels.
        """
        source_list = [
            el[0]["source"] for el in batch
        ]  # el[0] is the first half of a code snippet.
        # Following are the labels for the seq relationship loss: 0 -> negative pair, 1 -> positive pair.
        seq_relationship_labels = torch.randint(0, 2, (len(batch),)).long()
        target_list = [
            # seq_relationship_label==1 -> positive pair -> we take the second half of the code snippet
            # seq_relationship_label==0 -> negative pair -> we take a random code snippet given in el[1]
            el[0]["target"] if seq_relationship_labels[i] == 1 else el[1]["source"]
            for i, el in enumerate(batch)
        ]

        input_examples_list = [  # Combine source and target w/ template: [CLS] SOURCE [SEP] [TARGET] [SEP]
            f"{CLS_TOKEN}{source_list[i]}{SEPARATOR_TOKEN}{target_list[i]}{SEPARATOR_TOKEN}"
            for i in range(len(batch))
        ]

        positive_examples_list = [  # Positve example are perturbed versions of the source, used for the contrastive loss.
            f"{CLS_TOKEN}{perturb_words(source_list[i], self.contrastive_masking_probability, self.tokenizer.mask_token)}{SEPARATOR_TOKEN}"
            for i in range(len(batch))
        ]

        input_examples_encoding = self.tokenizer(
            input_examples_list,
            padding="longest",
            max_length=self.maximum_length,
            truncation=True,
            return_tensors="pt",
        )

        input_examples_ids = input_examples_encoding.input_ids
        input_examples_att_mask = (
            input_examples_encoding.attention_mask
        )  # Padding masks.
        input_examples_pooling_mask = get_pooling_mask(
            input_examples_ids, self.sep_token_id
        )  # Pooling masks indicate the first [SEP] occurrence, used for seq embedding.
        input_examples_ids, mlm_labels = perturb_tokens(
            input_examples_ids,
            input_examples_att_mask,
            self.mlm_masking_probability,
            self.mask_token_id,
        )  # Dynamically perturbs input tokens and generates corresponding mlm labels.

        positive_examples_encoding = self.tokenizer(
            positive_examples_list,
            padding="longest",
            max_length=self.maximum_length,
            truncation=True,
            return_tensors="pt",
        )

        positive_examples_ids = positive_examples_encoding.input_ids
        # Padding and pooling masks coincide for positive examples since there's only one [SEP] at the end.
        positive_examples_att_mask = positive_examples_encoding.attention_mask

        return (
            input_examples_ids,
            input_examples_att_mask,
            input_examples_pooling_mask,
            positive_examples_ids,
            positive_examples_att_mask,
            mlm_labels,
            seq_relationship_labels,
        )
