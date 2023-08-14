from typing import Dict, Iterator, List, Tuple, Union, cast

from transformers import is_datasets_available, requires_backends
from transformers.tokenization_utils_base import PreTokenizedInput, PreTokenizedInputPair, TextInput, TextInputPair

if is_datasets_available():
    from datasets import Dataset, DatasetDict
else:
    Dataset = None
    DatasetDict = None


def get_texts_from_dataset(
    dataset: Union[Dataset, DatasetDict], key: Union[str, Tuple[str, str]]
) -> Iterator[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]]:
    """Extract the texts of a dataset given their keys.

    Note:
        This function is only available if the `datasets`-extra is installed.

    Args:
        dataset: The huggingface dataset used for training.
        key: Either a simple string, being the key referring the text or a Tuple of two strings,
            referring to the keys for a text pair

    Returns: the texts or text pairs extracted from the dataset

    """
    requires_backends("get_texts_from_dataset", "datasets")

    datasets = [dataset] if isinstance(dataset, Dataset) else list(dataset.values())

    for ds in datasets:
        for ex in ds:
            ex = cast(Dict[str, Union[str, List[str]]], ex)
            if isinstance(key, str):
                yield ex[key]
            else:
                seq_a, seq_b = key
                yield ex[seq_a], ex[seq_b]
