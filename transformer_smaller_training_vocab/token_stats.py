from collections.abc import Sequence
from typing import Union

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PreTokenizedInput, PreTokenizedInputPair, TextInput, TextInputPair


def get_token_stats(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
) -> list[int]:
    used = {token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special}
    for text in texts:
        if isinstance(text, tuple):
            encoding = tokenizer(text[0], text[1], is_split_into_words=isinstance(text[0], list))
        else:
            encoding = tokenizer(text, is_split_into_words=isinstance(text, list))
        used.update(encoding["input_ids"])

    return sorted(used)
