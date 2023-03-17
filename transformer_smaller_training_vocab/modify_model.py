from typing import List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from transformers import PreTrainedModel


def reduce_embedding(
    model: PreTrainedModel,
    keep_token_ids: List[int],
    empty_cuda_cache: bool = True,
    optimizer: Optional[Optimizer] = None,
) -> torch.Tensor:
    model_device = model.device
    if optimizer is not None:
        found_param_group = None
        param_group_idx = -1
        for i, param_group in enumerate(optimizer.param_groups):
            if any(p is model.get_input_embeddings().weight for p in param_group["params"]):
                found_param_group = param_group
                param_group_idx = i
    else:
        found_param_group = None
        param_group_idx = -1

    if found_param_group is not None:
        del found_param_group["params"][param_group_idx]
    freeze = not model.get_input_embeddings().weight.requires_grad
    embedding_weights: torch.Tensor = model.get_input_embeddings().weight.detach().cpu()
    model.get_input_embeddings().__delattr__("weight")
    if empty_cuda_cache:
        torch.cuda.empty_cache()
    keep_tensor = torch.LongTensor(keep_token_ids)
    keep_embedding_weights = embedding_weights[keep_tensor]
    new_in_emb = nn.Embedding.from_pretrained(keep_embedding_weights, freeze=freeze)  # type: ignore[no-untyped-call]
    model.set_input_embeddings(new_in_emb)
    model.get_input_embeddings().to(model_device)
    model.config.vocab_size = keep_embedding_weights.size(0)
    if found_param_group is not None:
        found_param_group["params"].extend(model.get_input_embeddings().parameters())

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
