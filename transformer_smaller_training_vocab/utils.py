from typing import Union, Tuple, Iterator, cast, Dict, List

from datasets import Dataset, DatasetDict
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair


def get_texts_from_dataset(
    dataset: Union[Dataset, DatasetDict], key: Union[str, Tuple[str, str]]
) -> Iterator[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]]:
    if isinstance(dataset, Dataset):
        datasets = [dataset]
    else:
        datasets = list(dataset.values())

    for ds in datasets:
        for ex in ds:
            ex = cast(Dict[str, Union[str, List[str]]], ex)
            if isinstance(key, str):
                yield ex[key]
            else:
                seq_a, seq_b = key
                yield ex[seq_a], ex[seq_b]
