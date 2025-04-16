from transformers import PreTrainedTokenizerBase

from transformer_smaller_training_vocab.transformer_set_vocab import set_vocab


def reduce_tokenizer(tokenizer: PreTrainedTokenizerBase, used_token_ids: list[int]) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    reversed_vocab: dict[int, str] = dict(map(reversed, vocab.items()))  # type: ignore[arg-type]
    # typeignore -> map does not allow reversed

    reduced_vocab = {reversed_vocab[token_id]: i for i, token_id in enumerate(used_token_ids)}
    set_vocab(tokenizer, reduced_vocab)
    return vocab


def recreate_tokenizer(tokenizer: PreTrainedTokenizerBase, old_vocab: dict[str, int]) -> None:
    set_vocab(tokenizer, old_vocab)
