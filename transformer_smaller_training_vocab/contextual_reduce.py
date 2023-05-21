from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput, PreTokenizedInputPair, TextInput, TextInputPair

from transformer_smaller_training_vocab.logging_utils import logger
from transformer_smaller_training_vocab.modify_model import recreate_embedding, reduce_embedding
from transformer_smaller_training_vocab.modify_tokenizer import recreate_tokenizer, reduce_tokenizer
from transformer_smaller_training_vocab.token_stats import get_token_stats


def reduce_train_vocab_and_context(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
    empty_cuda_cache: Optional[bool] = None,
    optimizer: Optional[Optimizer] = None,
) -> Tuple[List[int], Dict[str, int], torch.Tensor]:
    """Reduce the vocabulary given a set of texts.

    Reduces the vocabulary of a model and a tokenizer by checking which tokens are used in the text
    and discarding all unused tokens.

    Args:
        model: The transformers model to reduce
        tokenizer: The tokenizer respective to the transformers model
        texts: A Sequence of either texts, pre-tokenized texts, text-pairs or pre-tokenized textpairs.
            Usually the full training + validation data used when training. The model & tokenizer vocabulary will be reduced
            to only tokens that are found in those texts.
        empty_cuda_cache: Defaults to True if the model is stored on cuda and False otherwise.
            If False, for some time, the weights will be in memory twice (Full + Reduced),
            before the garbage collection removes the Full weights from cache.
            If True, the cache will be emptied, before the reduced weights will be loaded to the device of the model and
            therefore won't have a temporarily higher memory footprint.
        optimizer: Defaults to None
            If provided, the optimizer parameters will be updated, to use the reduced embeddings instead of the old pointer.
            It is crucial to provide the optimizer if one was created before reducing the model.

    Returns:
        :All information required to restore the original vocabulary after training, consisting of:

        * **used_tokens** (List[int]) - The ids of all tokens that will be kept in vocabulary.
        * **saved_vocab** (Dict[str,int]) - The original vocabulary to recreate the tokenizer.
        * **saved_embeddings** (Tensor) - The original embedding weights.

    """
    if empty_cuda_cache is None:
        empty_cuda_cache = model.device.type == "cuda"
    initial_param_count = sum(p.numel() for p in model.parameters())
    initial_trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial_vocab_size = tokenizer.vocab_size
    logger.debug(f"Gathering token statistics for tokenizer {type(tokenizer)}")
    used_tokens = get_token_stats(tokenizer, texts)
    logger.info(f"Gathered {len(used_tokens)} of total {tokenizer.vocab_size}")
    saved_embeddings = reduce_embedding(model, used_tokens, optimizer=optimizer, empty_cuda_cache=empty_cuda_cache)
    logger.debug(f"Reduced embedding size for model {type(model)}")
    saved_vocab = reduce_tokenizer(tokenizer, used_tokens)
    logger.debug(f"Reduced tokenizer vocab {type(tokenizer)}")

    reduced_param_count = sum(p.numel() for p in model.parameters())
    reduced_trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduced_vocab_size = len(used_tokens)

    logger.info(f"Reducing vocab size by {1 - reduced_vocab_size / initial_vocab_size:.4%}")
    logger.info(f"Reducing model size by {1 - reduced_param_count / initial_param_count:.4%}")
    logger.info(
        f"Reducing training parameter count by {1 - reduced_trainable_param_count / initial_trainable_param_count:.4%}"
    )
    return used_tokens, saved_vocab, saved_embeddings


def recreate_vocab(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    used_tokens: List[int],
    saved_vocab: Dict[str, int],
    saved_embeddings: torch.Tensor,
    empty_cuda_cache: Optional[bool] = None,
) -> None:
    """Recreates the full vocabulary from a reduced model.

    Combines the stored embeddings with the updated embeddings of the reduced model and stores everything in place
    to have a model functioning on full vocabulary.

    Args:
        model: The reduced transformers model to recreate
        tokenizer: The reduced tokenizer to recreate
        used_tokens: The ids of tokens that are still contained
        saved_vocab: The full vocabulary that was saved.
        saved_embeddings: The saved embeddings of the full transformer before training.
        empty_cuda_cache: Defaults to True if the model is stored on cuda and False otherwise.
            If False, for some time, the weights will be in memory twice (Full + Reduced),
            before the garbage collection removes the Full weights from cache.
            If True, the cache will be emptied, before the reduced weights will be loaded to the device of the model and
            therefore won't have a temporarily higher memory footprint.
    """
    if empty_cuda_cache is None:
        empty_cuda_cache = model.device.type == "cuda"
    recreate_embedding(model, saved_embeddings, used_tokens, empty_cuda_cache=empty_cuda_cache)
    logger.debug(f"Recreated embedding size model {type(model)}")
    recreate_tokenizer(tokenizer, saved_vocab)
    logger.debug(f"Recreated tokenizer vocab model {type(model)}")


@contextmanager
def reduce_train_vocab(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
    empty_cuda_cache: Optional[bool] = None,
    optimizer: Optional[Optimizer] = None,
) -> Iterator[None]:
    """Contextmanager to temporary reduce the model for training.

    Examples:
        >>> with reduce_train_vocab(model, tokenizer, texts):
        >>>     # train reduced model
        >>> # save full model again

    Args:
        model: The transformers model to reduce
        tokenizer: The tokenizer respective to the transformers model
        texts: A Sequence of either texts, pre-tokenized texts, text-pairs or pre-tokenized textpairs.
            Usually the full training + validation data used when training. The model & tokenizer vocabulary will be reduced
            to only tokens that are found in those texts.
        empty_cuda_cache: Defaults to True if the model is stored on cuda and False otherwise.
            If False, for some time, the weights will be in memory twice (Full + Reduced),
            before the garbage collection removes the Full weights from cache.
            If True, the cache will be emptied, before the reduced weights will be loaded to the device of the model and
            therefore won't have a temporarily higher memory footprint.
        optimizer: Defaults to None
            If provided, the optimizer parameters will be updated, to use the reduced embeddings instead of the old pointer.
            It is crucial to provide the optimizer if one was created before reducing the model.
    """
    used_tokens, saved_vocab, saved_embeddings = reduce_train_vocab_and_context(
        model, tokenizer, texts, empty_cuda_cache, optimizer
    )

    yield

    recreate_vocab(model, tokenizer, used_tokens, saved_vocab, saved_embeddings, empty_cuda_cache=empty_cuda_cache)
