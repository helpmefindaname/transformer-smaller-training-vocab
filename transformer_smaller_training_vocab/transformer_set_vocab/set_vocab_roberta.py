from typing import Dict

from transformers import RobertaTokenizer

from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import register_set_vocab


@register_set_vocab(RobertaTokenizer)
def set_roberta_vocab(tokenizer: RobertaTokenizer, vocab: Dict[str, int]) -> None:
    tokenizer.encoder = vocab
    tokenizer.decoder = {v: k for k, v in vocab.items()}
    tokenizer.added_tokens_encoder = {k: vocab[k] for k in tokenizer.added_tokens_encoder.keys()}
