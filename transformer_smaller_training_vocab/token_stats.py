from typing import Union, List, Sequence

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair


def get_token_stats(
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
) -> List[int]:
    used = set()
    used.update(tokenizer.all_special_ids)
    for text in texts:
        if isinstance(text, tuple):
            encoding = tokenizer(text[0], text[1], is_split_into_words=isinstance(text[0], list))
        else:
            encoding = tokenizer(text, is_split_into_words=isinstance(text, list))
        used.update(encoding["input_ids"])

    return sorted(used)
