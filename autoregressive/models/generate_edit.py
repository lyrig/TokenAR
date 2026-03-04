# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

import pdb
### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True, **sampling_kwargs):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, input_txt_embs: torch.Tensor, input_img_indices: torch.Tensor, input_mode: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    '''
    input_txt_embs: (B, 120)
    input_img_indices: (B, 1024 * 4)
    '''
    if cfg_scale > 1.0:
        # logits, _ = model(None, cond_idx, input_pos)
        logits, _, _ = model(input_txt_embs=input_txt_embs, input_img_indices=input_img_indices, edited_img_indices=None, 
                             input_mode=input_mode, input_pos=input_pos, instruct_indices=sampling_kwargs.get("instruct_indices"))
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        # logits, _ = model(None, cond_idx, input_pos)
        logits, _, _ = model(input_txt_embs=input_txt_embs, input_img_indices=input_img_indices, edited_img_indices=None, \
                             input_mode=input_mode, input_pos=input_pos, block_len=sampling_kwargs.get("block_len"), \
                            num_ref=sampling_kwargs.get("num_ref"), instruct_indices=sampling_kwargs.get("instruct_indices"))

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _, _ = model(input_txt_embs=None, input_img_indices=None, edited_img_indices=x_combined, input_pos=input_pos, input_img_mask=sampling_kwargs.get("input_img_mask"))
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _, _ = model(input_txt_embs=None, input_img_indices=None, edited_img_indices=x, input_pos=input_pos, input_img_mask=sampling_kwargs.get("input_img_mask"))
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        if cfg_interval > -1 and i > cfg_interval:
            cfg_flag = False
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, input_txt_embs, input_img_indices, input_mode, max_new_tokens, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    '''
    Parameters:
        - model: autoregressive.models.gpt_edit.Transformer
        - input_txt_embs: (B, txt_token_length)
        - input_img_indices: (B, 1024)
        - 
    '''
    if model.model_type == 'edit':
        if cfg_scale > 1.0:
            input_txt_null = torch.zeros_like(input_txt_embs) + model.cls_embedding.uncond_embedding
            input_txt_combined = torch.cat([input_txt_embs, input_txt_null])
            input_img_combined = torch.cat([input_img_indices, input_img_indices])
            input_mode_combined = torch.cat([input_mode, input_mode])
        else:
            input_txt_combined = input_txt_embs
            input_img_combined = input_img_indices
            input_mode_combined = input_mode
        if sampling_kwargs.get("instruct_token_mode") != None:
            T = input_txt_combined.shape[1] + input_img_combined.shape[1] + sampling_kwargs.get("instruct_token_num", 0)
        else:
            T = input_txt_combined.shape[1] + input_img_combined.shape[1]
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = input_img_indices.shape[0]
    device = input_img_indices.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype, **sampling_kwargs)
   
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, input_txt_combined, input_img_combined, input_mode_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)
    if sampling_kwargs.get("return_all"):
        return seq
    return seq[:, T:]
