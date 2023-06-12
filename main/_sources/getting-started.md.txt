# Getting started

## Motivation

Have you ever trained a transformer model and noticed that most tokens in the vocab are not used?
Logically the token embeddings from those terms won't change, however they still take up memory and compute resources on your GPU.
One could assume that the embeddings are just a small part of the model and therefore aren't relevant, however looking at models like [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) have 45.72% of parameters as "word_embeddings".
Besides that, the gradient computation is done for the whole embedding weight, leading to gradient updates with very large amounts of 0s, eating a lot of memory, especially with state optimizers such as adam.

To reduce these inconveniences, this package provides a simple and easy to use way to
* gather usage statistics of the vocabulary
* temporary reduce the vocabulary to include no tokens that won't be used during training
* fit in the tokens back in after the training is finished, so the full version can be saved.


## Limitations

This library works fine, if you use any [FastTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
However if you want to use a `slow` tokenizer, it get's more tricky as huggingface-transformers has - per current date - no interface for overwriting the vocabulary in transformers.
So they require a custom implementation, currently the following tokenizers are supported:
* XLMRobertaTokenizer
* RobertaTokenizer
* BertTokenizer

## Requirements and Installation

The project is based on Transformers 4.1.0+, PyTorch 1.8+ and Python 3.7+
Then, in your favorite virtual environment, simply run:

```
pip install transformer-smaller-training-vocab
```

## Example Usage

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