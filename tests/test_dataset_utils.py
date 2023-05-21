from typing import Any, Dict

from datasets import load_dataset

from transformer_smaller_training_vocab import get_texts_from_dataset


def test_cola_dataset():
    count = 100

    def filter_by_id(example: Dict[str, Any]) -> bool:
        return example["idx"] < count

    dataset = load_dataset("glue", "cola").filter(filter_by_id)
    n = 0
    for text in get_texts_from_dataset(dataset, key="sentence"):
        assert isinstance(text, str)
        n += 1
    assert n == count * 3  # 3 splits


def test_ax_dataset():
    count = 100
    dataset = load_dataset("glue", "ax")["test"].select(list(range(count)))
    n = 0
    for text in get_texts_from_dataset(dataset, key=("premise", "hypothesis")):
        assert isinstance(text, tuple)
        assert len(text) == 2
        assert isinstance(text[0], str)
        assert isinstance(text[1], str)
        n += 1
    assert n == count * 1  # only test split


def test_cola_tokenized_dataset():
    count = 100

    def space_tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        example["tokens"] = example["sentence"].split()
        return example

    def filter_by_id(example: Dict[str, Any]) -> bool:
        return example["idx"] < count

    dataset = load_dataset("glue", "cola").filter(filter_by_id).map(space_tokenize)
    n = 0
    for text in get_texts_from_dataset(dataset, key="tokens"):
        assert isinstance(text, list)
        n += 1
    assert n == count * 3  # 3 splits


def test_ax_tokenized_dataset():
    def space_tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        example["tokens_a"] = example["premise"].split()
        example["tokens_b"] = example["hypothesis"].split()
        return example

    count = 100
    dataset = load_dataset("glue", "ax")["test"].select(list(range(count))).map(space_tokenize)
    n = 0
    for text in get_texts_from_dataset(dataset, key=("tokens_a", "tokens_b")):
        assert isinstance(text, tuple)
        assert len(text) == 2
        assert isinstance(text[0], list)
        assert isinstance(text[1], list)
        n += 1
    assert n == count * 1  # only test split
