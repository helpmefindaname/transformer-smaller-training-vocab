import collections

from transformers import BertTokenizer

from transformer_smaller_training_vocab.transformer_set_vocab.auto_set_vocab import register_set_vocab


@register_set_vocab(BertTokenizer)
def set_bert_vocab(tokenizer: BertTokenizer, vocab: dict[str, int]) -> None:
    tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in vocab.items()])
    tokenizer.vocab = vocab
    tokenizer.wordpiece_tokenizer.vocab = vocab
    tokenizer.added_tokens_decoder = {vocab[k]: k for k in tokenizer.added_tokens_encoder}
