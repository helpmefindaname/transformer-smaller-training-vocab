from typing import Callable, TypeVar

from transformers import PreTrainedTokenizerBase

TOK = TypeVar("TOK", bound=PreTrainedTokenizerBase)
TOK_FUNCTION = Callable[[TOK, dict[str, int]], None]

tokenizer_set_vocab_functions: dict[type[TOK], Callable[[TOK, dict[str, int]], None]] = {}  # type: ignore[valid-type]


def register_set_vocab(tokenizer_type: type[PreTrainedTokenizerBase]) -> Callable[[TOK_FUNCTION], TOK_FUNCTION]:
    def _decorator(fn: TOK_FUNCTION) -> TOK_FUNCTION:
        tokenizer_set_vocab_functions[tokenizer_type] = fn

        def _inner_decorator(tokenizer: TOK, vocab: dict[str, int]) -> None:
            fn(tokenizer, vocab)  # pragma: no cover  # coverage ignores this line

        return _inner_decorator

    return _decorator


def get_set_vocab_function(tokenizer_cls: type[PreTrainedTokenizerBase]) -> TOK_FUNCTION:
    set_vocab_function = tokenizer_set_vocab_functions.get(tokenizer_cls)
    if set_vocab_function is not None:
        return set_vocab_function
    for tokenizer_parent_cls, set_vocab_function in tokenizer_set_vocab_functions.items():
        if issubclass(tokenizer_cls, tokenizer_parent_cls):
            return set_vocab_function
    raise ValueError(f"type '{tokenizer_cls}' has no implementation for setting the vocabulary.")  # pragma: no cover


def set_vocab(tokenizer: PreTrainedTokenizerBase, vocab: dict[str, int]) -> None:
    tokenizer_cls = type(tokenizer)
    set_vocab_function = get_set_vocab_function(tokenizer_cls)
    set_vocab_function(tokenizer, vocab)
