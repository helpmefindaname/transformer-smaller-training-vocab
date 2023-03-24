from typing import Dict

from transformers import XLMRobertaTokenizer

from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import register_set_vocab


@register_set_vocab(XLMRobertaTokenizer)
def set_xlm_roberta_vocab(tokenizer: XLMRobertaTokenizer, vocab: Dict[str, int]) -> None:
    tokenizer.fairseq_tokens_to_ids = vocab
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    tokenizer.added_tokens_encoder = {k: vocab[k] for k in tokenizer.added_tokens_encoder.keys()}
