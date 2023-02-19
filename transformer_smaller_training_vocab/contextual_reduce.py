from contextlib import contextmanager
from typing import Union, Optional, Sequence, Iterator

from torch.optim import Optimizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair

from transformer_smaller_training_vocab.logging_utils import logger
from transformer_smaller_training_vocab.modify_model import reduce_embedding, recreate_embedding
from transformer_smaller_training_vocab.modify_tokenizer import recreate_tokenizer, reduce_tokenizer
from transformer_smaller_training_vocab.token_stats import get_token_stats


@contextmanager
def reduce_train_vocab(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
    empty_cuda_cache: Optional[bool] = None,
    optimizer: Optional[Optimizer] = None,
) -> Iterator[None]:
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

    yield

    recreate_embedding(model, saved_embeddings, used_tokens, empty_cuda_cache=empty_cuda_cache)
    logger.debug(f"Recreated embedding size model {type(model)}")
    recreate_tokenizer(tokenizer, saved_vocab)
    logger.debug(f"Recreated tokenizer vocab model {type(model)}")
