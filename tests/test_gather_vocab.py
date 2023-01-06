import itertools
from typing import Sequence, Union, List, Tuple, cast

from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, PreTokenizedInputPair, TextInputPair

from transformer_smaller_training_vocab.token_stats import get_token_stats


def calculate_and_assert_get_token_stats(
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
) -> Tuple[List[int], List[str]]:
    ids = get_token_stats(tokenizer, texts)

    assert len(set(ids)) == len(ids), "Ids are unique"
    assert ids == sorted(ids), "Ids are sorted by id"

    for special_id in tokenizer.all_special_ids:
        assert special_id in ids, "Expect to keep all special tokens"

    tokens = cast(List[str], tokenizer.convert_ids_to_tokens(ids))
    assert len(set(tokens)) == len(tokens), "Tokens are unique"

    return ids, tokens


def test_gather_texts_token_stats() -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = [
        "I live in Vienna",
        "George Washington was in Washington",
    ]
    ids, tokens = calculate_and_assert_get_token_stats(tokenizer, texts)

    input_ids = tokenizer(texts, is_split_into_words=False)["input_ids"]
    input_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
    for token in itertools.chain.from_iterable(input_tokens):
        assert token in tokens, "Tokens need to exist in vocab"


def test_gather_tokenized_token_stats() -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = [
        ["I", "live", "in", "Vienna"],
        ["George", "Washington", "was", "in", "Washington"],
    ]
    ids, tokens = calculate_and_assert_get_token_stats(tokenizer, texts)

    input_ids = tokenizer(texts, is_split_into_words=True)["input_ids"]
    input_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
    for token in itertools.chain.from_iterable(input_tokens):
        assert token in tokens, "Tokens need to exist in vocab"


def test_gather_text_pair_stats() -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = [
        ("I live in Vienna", "Where do I live ?"),
        ("George Washington was in Washington", "Where was GeorgeWashington?"),
    ]
    ids, tokens = calculate_and_assert_get_token_stats(tokenizer, texts)

    input_ids = tokenizer(*zip(*texts), is_split_into_words=False)["input_ids"]
    input_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
    for token in itertools.chain.from_iterable(input_tokens):
        assert token in tokens, "Tokens need to exist in vocab"


def test_gather_tokenized_text_pair_stats() -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = [
        (["I", "live", "in", "Vienna"], ["Where", "do", "I", "live", "?"]),
        (["George", "Washington", "was", "in", "Washington"], ["Where", "was", "George", "Washington", "?"]),
    ]
    ids, tokens = calculate_and_assert_get_token_stats(tokenizer, texts)

    input_ids = tokenizer(*zip(*texts), is_split_into_words=True)["input_ids"]
    input_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
    for token in itertools.chain.from_iterable(input_tokens):
        assert token in tokens, "Tokens need to exist in vocab"
