import json
from typing import List

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from transformer_smaller_training_vocab.modify_tokenizer import recreate_tokenizer, reduce_tokenizer
from transformer_smaller_training_vocab.token_stats import get_token_stats
from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import get_set_vocab_function

model_names = [
    "xlm-roberta-large",
    "bert-base-cased",
    "roberta-large",
]

fast_model_names = [
    ("microsoft/layoutlm-large-uncased", "WordPiece"),
    ("microsoft/layoutlm-base-cased", "BPE"),
    ("xlm-roberta-large", "Unigram"),
    ("sentence-transformers/all-mpnet-base-v2", "WordPiece"),
]
unsupported_tokenizers = ["google/electra-small-discriminator"]


def assert_reduction_and_creation_works(tokenizer: PreTrainedTokenizer, texts: List[str]) -> None:
    used_tokens = get_token_stats(tokenizer, texts)
    n = len(used_tokens)

    original_input_ids = tokenizer(texts)["input_ids"]
    original_token_texts = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in original_input_ids]

    assert any(
        any(i >= n for i in input_ids) for input_ids in original_input_ids
    ), "The example sentences need to cover ids that won't be used used at the end."
    assert all(
        all(i in used_tokens for i in input_ids) for input_ids in original_input_ids
    ), "All tokens need to be part of the used tokens"

    saved_vocab = reduce_tokenizer(tokenizer, used_tokens)

    reduced_input_ids = tokenizer(texts)["input_ids"]
    assert all(
        all(i < n for i in input_ids) for input_ids in reduced_input_ids
    ), "All reduced input ids need to be part of the vocabulary"
    recovered_input_ids = [[used_tokens[i] for i in input_ids] for input_ids in reduced_input_ids]
    reduced_token_texts = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in reduced_input_ids]

    assert (
        recovered_input_ids == original_input_ids
    ), "The reduced input_ids need to be translateable to the original vocab"

    assert reduced_token_texts == original_token_texts, "The respective output text needs to still be the same"

    recreate_tokenizer(tokenizer, saved_vocab)

    recreated_input_ids = tokenizer(texts)["input_ids"]
    recreated_token_texts = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in recreated_input_ids]
    assert recreated_input_ids == original_input_ids
    assert recreated_token_texts == original_token_texts


@pytest.mark.parametrize("model_name", model_names)
def test_slow_tokenizer_has_fixed_vocab(model_name: str) -> None:
    texts = [
        "I live in Vienna",
        "Home sweet home",
        "ay ay ay",
    ]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    assert_reduction_and_creation_works(tokenizer, texts)


@pytest.mark.parametrize("model_name, tokenizer_type", fast_model_names)
def test_fast_tokenizer_has_fixed_vocab(model_name: str, tokenizer_type: str) -> None:
    texts = [
        "I live in Vienna",
        "Home sweet home",
        "ay ay ay",
    ]
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
    tokenizer_model_type = tokenizer_json["model"]["type"]

    assert tokenizer_model_type == tokenizer_type, f"Expected to have a tokenizer of type '{tokenizer_type}'"

    assert_reduction_and_creation_works(tokenizer, texts)


@pytest.mark.parametrize("model_name", unsupported_tokenizers)
def test_tokenizer_is_not_supported_yet(model_name) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    with pytest.raises(ValueError):
        get_set_vocab_function(type(tokenizer))


@pytest.mark.parametrize("model_name, tokenizer_type", fast_model_names)
def test_fast_tokenizer_handles_special_tokens(model_name: str, tokenizer_type: str) -> None:
    texts = [
        "[FLERT] I live in Vienna [FLERT] Home sweet home",
    ]
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[FLERT]"]})
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
    tokenizer_model_type = tokenizer_json["model"]["type"]
    assert tokenizer_model_type == tokenizer_type, f"Expected to have a tokenizer of type '{tokenizer_type}'"

    assert_reduction_and_creation_works(tokenizer, texts)


@pytest.mark.parametrize("model_name", model_names)
def test_slow_tokenizer_handles_special_tokens(model_name: str) -> None:
    texts = [
        "[FLERT] I live in Vienna [FLERT] Home sweet home",
    ]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[FLERT]"]})
    assert_reduction_and_creation_works(tokenizer, texts)


def test_fast_word_level_tokenizer_has_fixed_vocab():
    texts = [
        "I live in Vienna",
        "Home sweet home",
        "ay ay ay",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "I": 0,
                    "live": 1,
                    "Hollywood": 2,
                    "New": 3,
                    "York": 4,
                    "Vienna": 5,
                    "Home": 6,
                    "sweet": 7,
                    "home": 8,
                    "ay": 9,
                    "[UNK]": 10,
                },
                unk_token="[UNK]",
            )
        )
    )
    assert_reduction_and_creation_works(tokenizer, texts)
