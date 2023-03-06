# transformer-smaller-training-vocab

[![PyPI version](https://badge.fury.io/py/transformer-smaller-training-vocab.svg)](https://badge.fury.io/py/transformer-smaller-training-vocab)
[![GitHub Issues](https://img.shields.io/github/issues/helpmefindaname/transformer-smaller-training-vocab.svg)](https://github.com/helpmefindaname/transformer-smaller-training-vocab/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Motivation

Have you ever trained a transformer model and noticed that most tokens in the vocab are not used?
Logically the token embeddings from those terms won't change, however they still take up memory and compute resources on your GPU.
One could assume that the embeddings are just a small part of the model and therefore aren't relevant, however looking at models like [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) have 45.72% of parameters as "word_embeddings".
Besides that, the gradient computation is done for the whole embedding weight, leading to gradient updates with very large amounts of 0s, eating a lot of memory, especially with state optimizers such as adam.

To reduce these inconveniences, this package provides a simple and easy to use way to
* gather usage statistics of the vocabulary
* temporary reduce the vocabulary to include no tokens that won't be used during training
* fit in the tokens back in after the training is finished, so the full version can be saved.


### Limitations

This library works fine, if you use any [FastTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
However if you want to use a `slow` tokenizer, it get's more tricky as huggingface-transformers has - per current date - no interface for overwriting the vocabulary in transformers.
So they require a custom implementation, currently the following tokenizers are supported:
* XLMRobertaTokenizer
* RobertaTokenizer
* BertTokenizer

If you want to use a tokenizer that is not on the list, please [create an issue](https://github.com/helpmefindaname/transformer-smaller-training-vocab/issues) for it.

## Quick Start

### Requirements and Installation

The project is based on Transformers 4.1.0+, PyTorch 1.8+ and Python 3.7+
Then, in your favorite virtual environment, simply run:

```
pip install transformer-smaller-training-vocab
```

### Example Usage

To use more efficient training, it is enough to do the following changes to an abitary training script:

```diff

  model = ...
  tokenizer = ...
  raw_datasets = ...
  ...

+ with reduce_train_vocab(model=model, tokenizer=tokenizer, texts=get_texts_from_dataset(raw_datasets, key="text")):
      def preprocess_function(examples):
          result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=True)
          result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
          return result
    
      raw_datasets = raw_datasets.map(
          preprocess_function,
          batched=True,
      )
    
      trainer = Trainer(
          model=model,
          train_dataset=raw_datasets["train"],
          eval_dataset=raw_datasets["validation"],
          tokenizer=tokenizer,
          ...
      )
    
      trainer.train()

+ trainer.save_model()  # save model at the end to contain the full vocab again.
```

Done! The Model will now be trained with only use the necessary parts of the token embeddings.

## Impact

Here is a table to document how much impact this technique has on training:

| **Model** | **Dataset** | **Vocab reduction** | **Model size reduction** |
|-----------|-------------|---------------------|--------------------------|
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) | CONLL 03 (en) |  93.13% | 42.58% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | CONLL 03 (en) | 93.13% | 64.31% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | CONLL 03 (en) | 43.64% | 08.97% |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased) | CONLL 03 (en) | 47.62% | 10.19% |
| [bert-large-uncased](https://huggingface.co/roberta-base) | CONLL 03 (en) | 47.62% | 04.44% |
| [roberta-base](https://huggingface.co/roberta-base) | CONLL 03 (en) | 58.39% | 18.08% |
| [roberta-large](https://huggingface.co/roberta-large) | CONLL 03 (en) | 58.39% | 08.45% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | cola | 77.665885% | 15.968120% |
| [roberta-base](https://huggingface.co/roberta-base) | cola | 86.081767% | 26.659724% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | cola | 97.785618% | 67.524955% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | mnli | 10.935991% | 2.248426% |
| [roberta-base](https://huggingface.co/roberta-base) | mnli | 14.781657% | 4.577886% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | mnli | 88.833689% | 61.343114% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | mrpc | 49.934474% | 10.266537% |
| [roberta-base](https://huggingface.co/roberta-base) | mrpc | 64.024669% | 19.828590% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | mrpc | 94.876841% | 65.516327% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | qnli | 8.618430% | 1.771951% |
| [roberta-base](https://huggingface.co/roberta-base) | qnli | 17.640505% | 5.463306% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | qnli | 87.566499% | 60.468238% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | qqp | 7.690716% | 1.581213% |
| [roberta-base](https://huggingface.co/roberta-base) | qqp | 5.908684% | 1.829933% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | qqp | 85.404117% | 58.975024% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | rte | 34.677197% | 7.129638% |
| [roberta-base](https://huggingface.co/roberta-base) | rte | 50.492390% | 15.637611% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | rte | 93.096055% | 64.286621% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | sst2 | 62.391364% | 12.827676% |
| [roberta-base](https://huggingface.co/roberta-base) | sst2 | 68.600418% | 21.245710% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | sst2 | 96.253630% | 66.467055% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | stsb | 51.345013% | 10.556619% |
| [roberta-base](https://huggingface.co/roberta-base) | stsb | 64.366856% | 19.934689% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | stsb | 94.881241% | 65.519546% |
| [bert-base-cased](https://huggingface.co/bert-base-cased) | wnli | 93.657746% | 19.256050% |
| [roberta-base](https://huggingface.co/roberta-base) | wnli | 96.033025% | 29.741652% |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | wnli | 99.253206% | 68.538385% |

Notice that while those reduced embeddings imply slightly less computation effort, those gains are neglectable, as the gradient computation for the parameters of transformer layers are dominant.
