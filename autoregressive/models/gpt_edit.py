# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
import math
import random
import pdb
def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'
    model_mode: Optional[str] = None
    distill_mode: Optional[Union[list, str]] = None

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    # New feature
    multi_cond: bool = False
    max_ref_num: int = 2
    max_edited_num: int = 1
    ref_index_embed: bool = False

    # New Feaeture(8.24): Instrcut Token(Impove A little. In Training Data Imporve A lot)
    # Combined with max_ref_num
    instruct_token_mode: Optional[str]=None
    instruct_token_num: int = 0

    # New Feature(8.28): Next Scale Prediction
    

    args = None



#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption) # type: ignore
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val # type: ignore
        v_out[:, :, input_pos] = v_val # type: ignore

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None,  # type: ignore
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv) # type: ignore
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output
    @torch.no_grad()
    def get_attention_map(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None,  # type: ignore
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv) # type: ignore
        else:
            keys, values = xk, xv
            
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        # --- 手动计算注意力 ---
        # 1. 计算注意力得分 (Query @ Key^T) / sqrt(d_k)
        attn_scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 2. 应用 mask (causal or provided)
        if mask is not None:
            attn_scores = attn_scores + mask
        elif seqlen > 1: # 应用因果 mask
            causal_mask = torch.triu(torch.ones_like(attn_scores), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # 3. 计算注意力权重 (Softmax)
        # 这就是我们需要的注意力图
        attention_weights = F.softmax(attn_scores, dim=-1)
        
        return attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, **kwargs):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, **kwargs))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_mode = config.model_mode
        self.distill_mode = config.distill_mode
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        # self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        # if self.model_type == 'c2i':
            # self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        # elif self.model_type == 't2i':
            # self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        if self.model_type == 'edit':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
            self.seq_length = self.block_size + self.cls_token_num + self.block_size
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim) # Image Token's feature map
        if config.multi_cond:
            self.register_buffer("img_uncond_embedding", nn.Parameter(torch.randn(self.block_size * config.max_ref_num, config.dim) / config.dim ** 0.5)) # New Feature
        else:
            self.register_buffer("img_uncond_embedding", nn.Parameter(torch.randn(self.block_size, config.dim) / config.dim ** 0.5))

        # New Feature: Reference Index Embedding
        if config.ref_index_embed:
            self.ref_index_embeddings = nn.Embedding(config.max_ref_num, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # New Feature: Instruct Token to instruct the generation content
        if config.instruct_token_num != 0:
            if config.instruct_token_mode == "special":
                self.instruct_token_embeddings = nn.Embedding(config.max_ref_num, config.instruct_token_num*config.dim)
            elif config.instruct_token_mode == "casual":
                self.instruct_token_embeddings = nn.Embedding(1, config.instruct_token_num*config.dim)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        if config.multi_cond:
            if config.max_ref_num <= 2:
                self.freqs_cis = precompute_freqs_cis_2d_edit2(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num) # type: ignore
            else:
                self.freqs_cis = precompute_freqs_cis_2d_editplus(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.config.max_ref_num, self.config.max_edited_num, self.config.instruct_token_num) # type: ignore
        else:
            self.freqs_cis = precompute_freqs_cis_2d_edit(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num) # type: ignore
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # alignment layer
        if isinstance(self.distill_mode, list):
            self.alignment = nn.ModuleList([nn.Conv2d(config.dim, 768, 2, 2) for distill_mode in self.distill_mode])
        else:
            if self.distill_mode == 'dinov2' or self.distill_mode == 'clip':
                # self.alignment = nn.Linear(config.dim, 768, bias=False)
                self.alignment = nn.Conv2d(config.dim, 768, 2, 2)
            if self.distill_mode == 'clipseg':
                self.alignment = nn.Conv2d(config.dim, 768, 2, 2)

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype, **setup_kwargs):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype) # type: ignore

        # New Feature: Change the mask princple
        if setup_kwargs.get("mask_mode") == None:
            causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
            self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        # New Feature: add speical mask format. citing: https://arxiv.org/pdf/2505.12274, the ICBP mask can improve a little.
        elif setup_kwargs.get("mask_mode") == "ICBP":
            causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
            block_len = setup_kwargs.get("block_len")
            num_ref = setup_kwargs.get("num_ref")   
            causal_mask[:num_ref * block_len] = False # type: ignore
            for _ in range(num_ref): # type: ignore
                causal_mask[_* block_len: (_ + 1) * block_len, _* block_len: (_ + 1) * block_len] = True # type: ignore
            self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        if self.config.multi_cond: # New Feature. 扩充了更多的位置编码用于Multiple Cond
            if self.config.max_ref_num <= 2:
                self.freqs_cis = precompute_freqs_cis_2d_edit2(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num) # type: ignore
            else:
                self.freqs_cis = precompute_freqs_cis_2d_editplus(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.config.max_ref_num, self.config.max_edited_num, self.config.instruct_token_num) # type: ignore
        else:
            self.freqs_cis = precompute_freqs_cis_2d_edit(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num) # type: ignore
        # 32, 64, 10000, 120

    def forward(
        self, 
        input_txt_embs: torch.Tensor,
        input_img_indices: torch.Tensor,
        edited_img_indices: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
        # New Feature
        input_img_mask: Optional[torch.Tensor] = None,
        instruct_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        
        ############## txt embedding dropout #################
        if self.model_mode == None:
            if (input_img_indices is not None) and (edited_img_indices is not None):
                # input ids
                input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training)[:,:self.cls_token_num]
                input_txt_embeddings = self.tok_dropout(input_txt_embeddings)
                input_img_embeddings = self.tok_embeddings(input_img_indices)
                edited_img_embeddings = self.tok_embeddings(edited_img_indices)
                token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, edited_img_embeddings), dim=1)[:, :-1]
                self.freqs_cis = self.freqs_cis.to(token_embeddings.device)

                targets = edited_img_indices
            else:
                if input_img_indices is not None: # prefill in inference
                    input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training)[:,:self.cls_token_num]
                    input_img_embeddings = self.tok_embeddings(input_img_indices)
                    token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings), dim=1)
                else: # decode_n_tokens(kv cache) in inference
                    token_embeddings = self.tok_embeddings(edited_img_indices)
                mask = self.causal_mask[:, None, input_pos]
        ############## joint embedding dropout #################
        if self.model_mode == 'joint_cls_emb':
            if (input_img_indices is not None) and (edited_img_indices is not None):
                B = input_txt_embs.shape[0]
                # input ids
                force_drop_ids = torch.rand(input_txt_embs.shape[0], device=input_txt_embs.device)
                force_img_drop_ids = (force_drop_ids < 0.1)
                force_txt_drop_ids = torch.logical_or((force_drop_ids < 0.05), (force_drop_ids > 0.95))
                
                input_img_embeddings = self.tok_embeddings(input_img_indices)
                # New Feature: add index embedding
                if input_img_mask != None and self.config.ref_index_embed:
                    input_index_embedding = self.ref_index_embeddings(input_img_mask)
                    input_img_embeddings += input_index_embedding

                input_img_embeddings = torch.where(force_img_drop_ids[:, None, None], self.img_uncond_embedding, input_img_embeddings) # type: ignore

                input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training, force_drop_ids=force_txt_drop_ids)[:,:self.cls_token_num]
                input_txt_embeddings = self.tok_dropout(input_txt_embeddings)

                edited_img_embeddings = self.tok_embeddings(edited_img_indices)
                # New Feature: Add Instruct token.
                if self.config.instruct_token_num != 0 and instruct_indices != None:
                    instruct_token_embeddings = self.instruct_token_embeddings(instruct_indices).reshape(-1, self.config.instruct_token_num, self.config.dim)
                    token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, instruct_token_embeddings, edited_img_embeddings), dim=1)[:, :-1].contiguous()
                elif self.config.instruct_token_num != 0:
                    instruct_token_embeddings = self.instruct_token_embeddings.weight[0].reshape(self.config.instruct_token_num, self.config.dim).unsqueeze(0).repeat(B, 1, 1) # Copy the First Channel
                    token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, instruct_token_embeddings, edited_img_embeddings), dim=1)[:, :-1].contiguous() # (B, 4096 + 120)
                else:
                    token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, edited_img_embeddings), dim=1)[:, :-1].contiguous() # (B, 4096 + 120)
                self.freqs_cis = self.freqs_cis.to(token_embeddings.device)

                targets = edited_img_indices
            else:
                if input_img_indices is not None: # prefill in inference
                    B = input_txt_embs.shape[0]
                    input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training)[:,:self.cls_token_num]
                    input_img_embeddings = self.tok_embeddings(input_img_indices)
                    # New Feature: add index embedding
                    if input_img_mask != None and self.config.ref_index_embed:
                        input_index_embedding = self.ref_index_embeddings(input_img_mask)
                        input_img_embeddings += input_index_embedding

                    if (input_mode==0).sum() > 0:
                        input_img_embeddings[input_mode==0] = self.img_uncond_embedding

                    # New Feature: Add Instruct tokens
                    if self.config.instruct_token_num != 0 and instruct_indices != None:
                        instruct_token_embeddings = self.instruct_token_embeddings(instruct_indices).reshape(-1, self.config.instruct_token_num, self.config.dim).repeat(B, 1, 1)
                        print(instruct_token_embeddings.shape)
                        token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, instruct_token_embeddings), dim=1) # (B, 4096 + 120)
                    elif self.config.instruct_token_num != 0:
                        instruct_token_embeddings = self.instruct_token_embeddings.weight[0].reshape(self.config.instruct_token_num, self.config.dim).unsqueeze(0).repeat(B, 1, 1) # Copy the First Channel
                        token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings, instruct_token_embeddings), dim=1) # (B, 4096 + 120)
                    else:
                        token_embeddings = torch.cat((input_img_embeddings, input_txt_embeddings), dim=1) # (B, 4096 + 120)
                    
                else: # decode_n_tokens(kv cache) in inference
                    token_embeddings = self.tok_embeddings(edited_img_indices) # (B, Length, channels)
                mask = self.causal_mask[:, None, input_pos]
        
        h = token_embeddings # type: ignore
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]] # type: ignore
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for _, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)

        # semantic features
        features = None
        if self.training and self.distill_mode is not None:
            features_lst = []
            if isinstance(self.distill_mode, list):
                for _i, distill_mode in enumerate(self.distill_mode):
                    features = h[:, -self.block_size:].view(h.shape[0], int(self.block_size**0.5), int(self.block_size**0.5), self.config.dim).clone()
                    features = features.permute(0, 3, 1, 2).contiguous()
                    features = self.alignment[_i](features) # type: ignore
                    if distill_mode == 'dinov2':
                        features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()
                    if distill_mode == 'clip':
                        features = F.interpolate(features, size=(14, 14), mode="bilinear", antialias=True, align_corners=False)
                        features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()
                    if distill_mode == 'clipseg':
                        features = F.interpolate(features, size=(22, 22), mode="bilinear", antialias=True, align_corners=False)
                        features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()
                    features_lst.append(features)
            else:
                features = h[:, -self.block_size:].view(h.shape[0], int(self.block_size**0.5), int(self.block_size**0.5), self.config.dim).clone()
                features = features.permute(0, 3, 1, 2).contiguous()
                features = self.alignment(features) # type: ignore
                if self.distill_mode == 'dinov2':
                    features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()
                if self.distill_mode == 'clip':
                    features = F.interpolate(features, size=(14, 14), mode="bilinear", antialias=True, align_corners=False)
                    features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()
                if self.distill_mode == 'clipseg':
                    features = F.interpolate(features, size=(22, 22), mode="bilinear", antialias=True, align_corners=False)
                    features = features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1).contiguous()

        logits = self.output(h).float()
        if hasattr(self, "intermediate_layers_feat"):
            self.intermediate_layers_feat.setdefault("logits", [])
            self.intermediate_layers_feat["logits"].append(logits.detach().cpu()) # type: ignore

        if self.training:
            if self.config.multi_cond:
                logits = logits[:, self.block_size * self.config.max_ref_num + self.cls_token_num + self.config.instruct_token_num - 1:].contiguous() # New feature 由于又bg所以需要两倍的blocksize
            else:
                logits = logits[:, self.block_size  + self.cls_token_num + self.config.instruct_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none') # type: ignore
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1) # type: ignore
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1) # type: ignore
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if isinstance(self.distill_mode, list):
            return logits, loss, features_lst # type: ignore
        return logits, loss, features


    @torch.no_grad()
    def get_intermediate_layers(
        self, 
        input_txt_embs: torch.Tensor,
        input_img_indices: torch.Tensor,
        input_mode: Optional[torch.Tensor] = None,
        edited_img_indices: Optional[torch.Tensor] = None,
        # New Feature
        max_new_token: Optional[int] = None,
        cfg_scale: float=3.0,
        cfg_interval: int=-1,
        return_cross_attention_map: bool = False,
        **sampling_kwargs
    )->Union[torch.Tensor, List]:
        self.eval()
        self.intermediate_layers_feat = {}
        handles = []
        target_layer = sampling_kwargs.get("n", None)
        if isinstance(target_layer, int):
            if return_cross_attention_map:
                for i in range(len(self.layers) - 1, max(len(self.layers) - target_layer - 1, -1), -1):
                    handles.append(self.layers[i].attention.register_forward_hook(self._cross_attn_map_forward_hook(f"layer_{i}"))) # type: ignore
            else:
                for i in range(len(self.layers) - 1, max(len(self.layers) - target_layer - 1, -1), -1):
                    handles.append(self.layers[i].register_forward_hook(self._layer_forward_hook(f"layer_{i}"))) # Save these hooks to remove
        elif isinstance(target_layer, list):
            if return_cross_attention_map:
                for i in target_layer:
                    handles.append(self.layers[i].attention.register_forward_hook(self._cross_attn_map_forward_hook(f"layer_{i}"))) # type: ignore
            else:
                for i in target_layer:
                    handles.append(self.layers[i].register_forward_hook(self._layer_forward_hook(f"layer_{i}"))) # Save these hooks to remove
        else:
            raise "Need N as List or int." # type: ignore
        
        if edited_img_indices == None:
            from autoregressive.models.generate_edit import generate
            output = generate(
                self,
                input_txt_embs,
                input_img_indices,
                input_mode,
                max_new_token,
                cfg_scale,
                cfg_interval,
                **sampling_kwargs
            ) # The Whole Sequence
            print(f"Generating Success.")
        else: 
            output = edited_img_indices
        
        output_embedding = self.tok_embeddings(output)
        # Remove these hooks
        for handle in handles:
            handle.remove()
        self.train()
        return (output, output_embedding, self.intermediate_layers_feat) # type: ignore 

    def _layer_forward_hook(self, name):
        assert hasattr(self, "intermediate_layers_feat")
        def hook(layer, input, output:torch.Tensor):
            self.intermediate_layers_feat.setdefault(name, [])
            self.intermediate_layers_feat[name].append(output.detach().cpu())
        return hook
    
    def _cross_attn_map_forward_hook(self, name):
        assert hasattr(self, "intermediate_layers_feat")
        def hook(module:Attention, input, output:torch.Tensor):
            self.intermediate_layers_feat.setdefault(name, [])
            self.intermediate_layers_feat[name].append(module.get_attention_map(*input).float().cpu())
        return hook


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 

# # 32, 64, 10000, 120
def precompute_freqs_cis_2d_edit(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    _grid_size = grid_size * 2 # 64
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(_grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, _grid_size, -1),
        freqs[None, :, :].expand(_grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (_grid_size, _grid_size, head_dim // 2, 2)
    # cache = cache_grid.flatten(0, 1)
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    cache_input_img = cache_grid[:grid_size, :grid_size].flatten(0, 1)
    cache_edit_img = cache_grid[-grid_size:, -grid_size:].flatten(0, 1)
    cond_cache = torch.cat([cache_input_img, torch.zeros(cls_token_num, n_elem // 2, 2), cache_edit_img]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache  # [2168, 32, 2]

# New Feature，更多的位置编码
def precompute_freqs_cis_2d_edit2(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    _grid_size = grid_size * 4 # 64
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(_grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, _grid_size, -1),
        freqs[None, :, :].expand(_grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    # cache = cache_grid.flatten(0, 1)
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    cache_input_img = cache_grid[:grid_size, :grid_size * 2].flatten(0, 1)
    cache_edit_img = cache_grid[-grid_size:, -grid_size:].flatten(0, 1)
    cond_cache = torch.cat([cache_input_img, torch.zeros(cls_token_num, n_elem // 2, 2), cache_edit_img]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache  # [2168, 32, 2]

# New Feature，更多的位置编码
def precompute_freqs_cis_2d_editplus(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, ref_num:int=4, edited_num:int=1, instruct_token_num=0, laplace_level:int=1):
    import math
    grid_m = int(math.sqrt(ref_num))
    grid_e = int(max(1, math.sqrt(edited_num)))
    # split the dimension into half, one for x and one for y
    _grid_size = grid_size * (grid_m + grid_e) # 64
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(_grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, _grid_size, -1),
        freqs[None, :, :].expand(_grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    # cache = cache_grid.flatten(0, 1)
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    cache_input_img = cache_grid[:grid_size * grid_m, :grid_size * grid_m].flatten(0, 1)
    cache_edit_img = cache_grid[-grid_size * grid_e:, -grid_size*grid_e:].flatten(0, 1)
    if instruct_token_num != 0 and instruct_token_num != None:
        if isinstance(instruct_token_num, int):
            pass
        else:
            instruct_token_num = int(instruct_token_num)
        cond_cache = torch.cat([cache_input_img, torch.zeros(cls_token_num, n_elem // 2, 2), torch.zeros(instruct_token_num, n_elem // 2, 2), cache_edit_img]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    else:
        cond_cache = torch.cat([cache_input_img, torch.zeros(cls_token_num, n_elem // 2, 2), cache_edit_img]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache  # [2168, 32, 2]


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}