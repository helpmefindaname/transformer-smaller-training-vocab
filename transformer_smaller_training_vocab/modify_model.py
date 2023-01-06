from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel


def reduce_embedding(model: PreTrainedModel, keep_token_ids: List[int], empty_cuda_cache: bool = True) -> torch.Tensor:
    model_device = model.device
    embedding_weights: torch.Tensor = model.get_input_embeddings().weight.detach().cpu()
    model.get_input_embeddings().__delattr__("weight")
    if empty_cuda_cache:
        torch.cuda.empty_cache()
    keep_tensor = torch.LongTensor(keep_token_ids)
    keep_embedding_weights = embedding_weights[keep_tensor]
    new_input_embedding = nn.Embedding(
        keep_embedding_weights.size(0), keep_embedding_weights.size(1), _weight=keep_embedding_weights
    )
    model.set_input_embeddings(new_input_embedding)
    model.get_input_embeddings().to(model_device)

    return embedding_weights


def recreate_embedding(
    model: PreTrainedModel, saved_embeddings: torch.Tensor, keep_token_ids: List[int], empty_cuda_cache: bool = True
) -> None:
    model_device = model.device
    embedding_weights: torch.Tensor = model.get_input_embeddings().cpu().weight.detach()
    model.get_input_embeddings().__delattr__("weight")
    if empty_cuda_cache:
        torch.cuda.empty_cache()  # pragma: no cover  # no need to test this line
    for reduced_id, full_id in enumerate(keep_token_ids):
        saved_embeddings[full_id] = embedding_weights[reduced_id]
    new_input_embedding = nn.Embedding(saved_embeddings.size(0), saved_embeddings.size(1), _weight=saved_embeddings)
    model.set_input_embeddings(new_input_embedding)
    model.get_input_embeddings().to(model_device)
