import json
from typing import Dict, Callable, Any

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import register_set_vocab


SET_VOCAB_FUNCTION = Callable[[Dict[str, Any], Dict[str, int]], None]

vocab_functions: Dict[str, SET_VOCAB_FUNCTION] = {}


def register_vocab_function(name: str) -> Callable[[SET_VOCAB_FUNCTION], SET_VOCAB_FUNCTION]:
    def _decorator(fn: SET_VOCAB_FUNCTION) -> SET_VOCAB_FUNCTION:
        vocab_functions[name] = fn

        def _inner_decorator(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
            fn(tokenizer_obj, vocab)  # pragma: no cover  # coverage ignores this line

        return _inner_decorator

    return _decorator


@register_set_vocab(PreTrainedTokenizerFast)
def set_fast_tokenizer_vocab(tokenizer: PreTrainedTokenizerFast, vocab: Dict[str, int]) -> None:

    tokenizer_obj = json.loads(tokenizer.backend_tokenizer.to_str())
    fix_special_tokens(tokenizer_obj, vocab)
    vocab_functions[tokenizer_obj["model"]["type"]](tokenizer_obj, vocab)
    json_data = json.dumps(tokenizer_obj)
    tokenizer._tokenizer = Tokenizer.from_str(json_data)


def fix_special_tokens(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
    for special_token in tokenizer_obj["added_tokens"]:
        special_token["id"] = vocab[special_token["content"]]
    post_obj = tokenizer_obj["post_processor"]
    if post_obj is None or "special_tokens" not in post_obj:
        return
    for k in post_obj["special_tokens"]:
        post_obj["special_tokens"][k]["ids"] = [vocab[t] for t in post_obj["special_tokens"][k]["tokens"]]


@register_vocab_function("Unigram")
def set_unigram_vocab(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
    # unigram tokenizer save scores next to the vocabulary and requires other tokens for intermediate tokenization.
    # hence we don't delete tokens, but reorder them, so that the ids fit the requirement.
    n = len(vocab)

    # special tokens will be added to the end (and therefore have a wrong id)
    # if they are not explicitely also in the vocab.
    registered_tokens = {token for token, _ in tokenizer_obj["model"]["vocab"]}
    for special_token in tokenizer_obj["added_tokens"]:
        if special_token["content"] not in registered_tokens:
            tokenizer_obj["model"]["vocab"].append((special_token["content"], 0.0))

    tokenizer_obj["model"]["vocab"] = sorted(tokenizer_obj["model"]["vocab"], key=lambda x: vocab.get(x[0], n))


@register_vocab_function("WordPiece")
def set_wordpiece_vocab(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
    tokenizer_obj["model"]["vocab"] = vocab


@register_vocab_function("WordLevel")
def set_wordlevel_vocab(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
    tokenizer_obj["model"]["vocab"] = vocab


@register_vocab_function("BPE")
def set_bpe_vocab(tokenizer_obj: Dict[str, Any], vocab: Dict[str, int]) -> None:
    # bpe tokenizer save merges next to the vocabulary and requires tokens for correct validation.
    # hence we don't delete tokens, but reorder them, so that the ids fit the requirement.
    old_vocab = tokenizer_obj["model"]["vocab"]

    new_vocab = vocab
    for k in old_vocab.keys():
        if k not in new_vocab:
            new_vocab[k] = len(new_vocab)

    tokenizer_obj["model"]["vocab"] = new_vocab
