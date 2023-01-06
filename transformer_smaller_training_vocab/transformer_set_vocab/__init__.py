from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import register_set_vocab, set_vocab
from transformer_smaller_training_vocab.transformer_set_vocab.set_vocab_bert import set_bert_vocab
from transformer_smaller_training_vocab.transformer_set_vocab.set_vocab_fast_tokenizer import set_fast_tokenizer_vocab
from transformer_smaller_training_vocab.transformer_set_vocab.set_vocab_roberta import set_roberta_vocab
from transformer_smaller_training_vocab.transformer_set_vocab.set_vocab_xlm_roberta import set_xlm_roberta_vocab


__all__ = [
    "register_set_vocab",
    "set_vocab",
    "set_xlm_roberta_vocab",
    "set_roberta_vocab",
    "set_bert_vocab",
    "set_fast_tokenizer_vocab",
]
