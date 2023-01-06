import importlib

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from transformer_smaller_training_vocab import reduce_train_vocab


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("importlib") is None,
    reason="Can only measure gpu memory when a gpu is available",
)
def test_reduced_memory():
    model_name = "distilbert-base-uncased"

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to("cuda:0")
    texts = [
        "I live in Vienna",
        "Home sweet home",
        "ay ay ay",
    ]

    full_model = torch.cuda.memory_reserved()

    with reduce_train_vocab(model=model, tokenizer=tokenizer, texts=texts, empty_cuda_cache=True):
        reduced_model = torch.cuda.memory_reserved()

    full_again_model = torch.cuda.memory_reserved()

    assert (
        full_model == full_again_model
    ), "As no other parameters are set, the gpu memory should be the same before the reduction and after finishing."

    assert (
        reduced_model < full_model
    ), "During the reduction we expect to use less gpu memory, as we are freeing the embedding full embedding matrix."

    assert (
        1 - reduced_model / full_model > 0.30
    ), "In the case of distilbert, we expect more than 30% memory reduction with our (very small) vocab."


def test_reduction_works():
    model_name = "distilbert-base-uncased"

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        "I live in Vienna",
        "Home sweet home",
        "ay ay ay",
    ]

    with reduce_train_vocab(model=model, tokenizer=tokenizer, texts=texts):
        pass
