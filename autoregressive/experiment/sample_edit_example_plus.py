import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import pdb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image

import time
import argparse
import random
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt_edit import GPT_models
from autoregressive.models.generate_edit import generate
# from dataset.build import build_dataset
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
import glob
import json
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from autoregressive.sample.metrics_calculator import MetricsCalculator

import csv

from dataset.Condition_MultiGen_Depth_eval import MultiGen_Depth_Eval_Dataset
from dataset.Condition_MultiGen_Canny_eval import MultiGen_Canny_Eval_Dataset
from dataset.Condition_Segmentation_eval import Condition_Segmentation_Eval_Dataset
from dataset.Condition_MultiCond_IP_plus import MultiCond_IP_Plus_Dataset
from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline, # type: ignore
    StableDiffusionControlNetPipeline, ControlNetModel, # type: ignore
    UniPCMultistepScheduler, DDIMScheduler, # type: ignore
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, # type: ignore
    StableDiffusionXLControlNetPipeline, AutoencoderKL # type: ignore
)


def main(args):
    random.seed(args.seed)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = args.device if torch.cuda.is_available() else "cpu"
    need_bg = args.multi_cond
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")
    writer = SummaryWriter(args.log_dir)
    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    latent_size = args.image_size // args.downsample_size # 512 / 16 = 32
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        model_mode=args.gpt_mode,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        distill_mode=args.distill_mode,
        multi_cond=need_bg,
        max_ref_num=args.max_ref_num, 
        ref_index_embed=args.add_ref_embed, 
        max_edited_num=args.max_ref_num if (args.concat_target and args.instruct_token_mode in [None, "casual"]) else 1,
        instruct_token_num = args.instruct_token_num if args.instruct_token_mode in ["casual", "special"] else 0,
        instruct_token_mode = args.instruct_token_mode
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # save_path = './examples'
    data_path = args.dataset
    # data_path = "your_path/data/subject_dataset_10k/eval"

    dataset = MultiCond_IP_Plus_Dataset(args, dataset_path=data_path, llm_tokenizer=t5_model.tokenizer, mode='val', need_bg=need_bg, max_cond=args.max_ref_num)

    def collate_fn(examples):
        # This will be a list of dictionaries, where each dictionary is one data sample from __getitem__
        
        # Let's check if the 'bg_img' key exists in the first example to handle both cases
        has_bg_img = 'bg_img' in examples[0]

        # Batching the text tokens and masks
        input_ids = torch.stack([e['input_ids'] for e in examples])
        input_ids_attn_mask = torch.stack([e['input_ids_attn_mask'] for e in examples])

        # Batching the edited images
        edited_img = torch.stack([e['edited_img'] for e in examples])
        
        # Batching the input images. The 'input_img' is already a list of tensors, so we need to stack them first
        # and then concatenate the batch.
        input_img = []
        for i in range(len(examples[0]["input_img"])):
            input_img.append(torch.stack([e['input_img'][i] for e in examples]))

        # Collecting the other data

        index = [e['index'] for e in examples]
        dataset = [e['dataset'] for e in examples]
        path = [e['path'] for e in examples]
        mode = torch.tensor([e['mode'] for e in examples], dtype=torch.int)
        instruct = torch.tensor([e['instruct'] for e in examples], dtype=torch.int)

        # Creating the final dictionary
        batch = {
            'index': index,
            'dataset': dataset,
            'mode': mode,
            'input_ids': input_ids,
            'input_ids_attn_mask': input_ids_attn_mask,
            'input_img': input_img,
            'edited_img': edited_img,
            'path': path,
            "instruct": instruct
        }

        # If the 'bg_img' exists, add it to the batch
        if has_bg_img:
            bg_img = torch.stack([e['bg_img'] for e in examples])
            batch['bg_img'] = bg_img

        return batch

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )


    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating Processing:"):
        if args.max_ref_num <= 1:
            # input_img, edited_img, target_ids, input_ids, prompts
            input_img = batch['input_img'].to(device, non_blocking=True)
            # New Feature
            if need_bg:
                bg_img = batch['bg_img'].to(device, non_blocking=True)
            # edited_img = batch['edited_img'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
            input_mode = batch['mode'].to(device, non_blocking=True)
            save_path = batch["path"][0]

            # process text ids to embeddings
            with torch.no_grad():
                input_txt_embs = t5_model.model(
                    input_ids=input_ids,
                    attention_mask=input_ids_attn_mask,
                )['last_hidden_state'].detach() # (B, length, channels)
            
            # process image ids to embeddings
            with torch.no_grad():
                _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                input_img_indices = input_img_indices.reshape(input_img.shape[0], -1) # (B, token_length)

                if need_bg:
                    _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                    bg_img_indices = bg_img_indices.reshape(bg_img.shape[0], -1) # type: ignore # (B, token_length)
                    input_img_indices = torch.concat([input_img_indices, bg_img_indices], dim=1)
            # pdb.set_trace()
            B = input_img.shape[0]
            qzshape = [B, args.codebook_embed_dim, latent_size, latent_size]
            t1 = time.time()
            index_sample = generate(
                gpt_model, input_txt_embs, input_img_indices, input_mode, latent_size ** 2, 
                emb_masks=None, 
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                
                )
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
            
            t2 = time.time()
            samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            assert samples.shape[0] == 1
            decoder_time = time.time() - t2
            print(f"decoder takes about {decoder_time:.2f} seconds.")
            ckpt_num = os.path.basename(args.gpt_ckpt).split(".")[0]
            final_save_path = f"{save_path}/{batch['dataset'][0]}_sample_txt_{args.cfg_scale}_multi_cond_test_{ckpt_num}.png"
            save_image(samples, final_save_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"image is saved to {final_save_path}")
        else:
            # input_img, edited_img, target_ids, input_ids, prompts
            # bg_img = batch["bg_img"].to(device, non_blocking=True)
            input_img_list = batch['input_img']
            input_img_list = list(map(lambda x: x.to(device, non_blocking=True), input_img_list))
            # New Feature
            if need_bg:
                bg_img = batch['bg_img'].to(device, non_blocking=True)
            # edited_img = batch['edited_img'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
            input_mode = batch['mode'].to(device, non_blocking=True)
            save_path = batch["path"][0]
            instruct_indices = batch['instruct'].to(device, non_blocking=True)

            # process text ids to embeddings
            with torch.no_grad():
                input_txt_embs = t5_model.model(
                    input_ids=input_ids,
                    attention_mask=input_ids_attn_mask,
                )['last_hidden_state'].detach() # (B, length, channels)
            
            # process image ids to embeddings
            with torch.no_grad():
                _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                # bg_img_indices = bg_img_indices.reshape(bg_img.shape[0], -1) # type: ignore # (B, token_length)
                # _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                # input_img_indices = input_img_indices.reshape(input_img.shape[0], -1) # (B, token_length)
                ref_index_mask = []
                input_img_indices_lst = []
                for i, input_img in enumerate(input_img_list): # type: ignore
                    _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                    
                    input_img_indices_lst.append(input_img_indices.reshape(input_img.shape[0], -1))
                    ref_index_mask.append((torch.ones_like(input_img_indices_lst[-1], dtype=torch.int) * i).to(input_img_indices_lst[-1]))\
                
                # Add Background
                # _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                input_img_indices_lst.append(bg_img_indices.reshape(input_img.shape[0], -1)) # type: ignore
                ref_index_mask.append((torch.ones_like(input_img_indices_lst[-1], dtype=torch.int) * i).to(input_img_indices_lst[-1])) # type: ignore
                input_img_indices = torch.concat(input_img_indices_lst, dim=1)
                input_img_mask = torch.concat(ref_index_mask, dim=1)

                    # input_img_indices = torch.concat([input_img_indices, bg_img_indices], dim=1)
            
            # pdb.set_trace()
            B = input_img_mask.shape[0]
            if args.concat_target and args.instruct_token_mode != "special":
                qzshape = [B, args.codebook_embed_dim, latent_size, latent_size] # The model prediction contains the reference part
                max_token = (latent_size ** 2) * args.max_ref_num
            else:
                qzshape = [B, args.codebook_embed_dim, latent_size, latent_size]
                max_token = latent_size**2
            t1 = time.time()
            
            # 实验证明该Mask方式存在极大的生成问题
            if args.special_mask:
                index_sample = generate(
                    gpt_model, input_txt_embs, input_img_indices, input_mode, max_token, 
                    emb_masks=None, 
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True, 
                    # New Feature: Add ref index mask and activate special mask.
                    mask_mode="ICBP",
                    block_len=input_img_indices_lst[0].shape[1],
                    num_ref=len(input_img_indices_lst),

                    # New Feature: Add Index Embedding
                    input_img_mask=input_img_mask,
                    device=device
                    )
            elif args.add_ref_embed:
                if args.instruct_token_mode == "special":
                    index_sample = generate(
                        gpt_model, input_txt_embs, input_img_indices, input_mode, max_token, 
                        emb_masks=None, 
                        cfg_scale=args.cfg_scale,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 

                        # New Feature: Add Index Embedding
                        input_img_mask=input_img_mask,
                        instruct_indices=instruct_indices,
                        device=device,
                        instruct_token_mode=args.instruct_token_mode,
                        instruct_token_num=args.instruct_token_num
                        )
                else:
                    index_sample = generate(
                        gpt_model, input_txt_embs, input_img_indices, input_mode, max_token, 
                        emb_masks=None, 
                        cfg_scale=args.cfg_scale,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 

                        # New Feature: Add Index Embedding
                        input_img_mask=input_img_mask,
                        device=device,
                        instruct_token_mode=args.instruct_token_mode,
                        instruct_token_num=args.instruct_token_num
                        )
            else:
                index_sample = generate(
                    gpt_model, input_txt_embs, input_img_indices, input_mode, max_token, 
                    emb_masks=None, 
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True, 
                    device=device
                    )

            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
            # pdb.set_trace()
            t2 = time.time()
            if args.concat_target and args.instruct_token_mode != "special":
                for i in range(args.max_ref_num):
                    tmp_sample = index_sample[:, i*(latent_size**2):(i+1)*(latent_size**2)]    
                    samples = vq_model.decode_code(tmp_sample, qzshape) # output value is between [-1, 1]
                    writer.add_image(f"Image_{i}", samples[0], global_step=0)
            else:
                samples = vq_model.decode_code(index_sample, qzshape)

            decoder_time = time.time() - t2
            print(f"decoder takes about {decoder_time:.2f} seconds.")
            ckpt_num = os.path.basename(args.gpt_ckpt).split(".")[0]
            final_save_path = f"{save_path}/sample_txt_{args.cfg_scale}_multi_cond_test_{ckpt_num}.png" if args.additional_info == None else f"{save_path}/{args.additional_info}_sample_txt_{args.cfg_scale}_multi_cond_test_{ckpt_num}.png"
            save_image(samples, final_save_path, nrow=4, normalize=True, value_range=(-1, 1)) # type: ignore
            print(f"image is saved to {final_save_path}")
            

@torch.no_grad()
def training_generate(timestep, gpt_model, vq_model, t5_model, loader, args):
    gpt_model.eval()
    vq_model.eval()
    t5_model.eval()
    device = args.device if torch.cuda.is_available() else "cpu"
    need_bg = args.multi_cond

    # create and load gpt model
    latent_size = args.image_size // args.downsample_size # 512 / 16 = 32


    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating Processing:"):
        if args.max_ref_num <= 1:
            # input_img, edited_img, target_ids, input_ids, prompts
            input_img = batch['input_img'].to(device, non_blocking=True)
            # New Feature
            if need_bg:
                bg_img = batch['bg_img'].to(device, non_blocking=True)
            # edited_img = batch['edited_img'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
            input_mode = batch['mode'].to(device, non_blocking=True)
            save_path = batch["path"][0]

            # process text ids to embeddings
            with torch.no_grad():
                input_txt_embs = t5_model.model(
                    input_ids=input_ids,
                    attention_mask=input_ids_attn_mask,
                )['last_hidden_state'].detach() # (B, length, channels)
            
            # process image ids to embeddings
            with torch.no_grad():
                _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                input_img_indices = input_img_indices.reshape(input_img.shape[0], -1) # (B, token_length)

                if need_bg:
                    _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                    bg_img_indices = bg_img_indices.reshape(bg_img.shape[0], -1) # type: ignore # (B, token_length)
                    input_img_indices = torch.concat([input_img_indices, bg_img_indices], dim=1)
            # pdb.set_trace()
            B = input_img.shape[0]
            qzshape = [B, args.codebook_embed_dim, latent_size, latent_size]
            t1 = time.time()
            index_sample = generate(
                gpt_model, input_txt_embs, input_img_indices, input_mode, latent_size ** 2, 
                emb_masks=None, 
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                )
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
            
            t2 = time.time()
            samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            assert samples.shape[0] == 1
            decoder_time = time.time() - t2
            print(f"decoder takes about {decoder_time:.2f} seconds.")
            ckpt_num = os.path.basename(args.gpt_ckpt).split(".")[0]
            final_save_path = f"{save_path}/{batch['dataset'][0]}_sample_txt_{args.cfg_scale}_multi_cond_test_{ckpt_num}.png"
            save_image(samples, final_save_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"image is saved to {final_save_path}")
        else:
            # input_img, edited_img, target_ids, input_ids, prompts
            # bg_img = batch["bg_img"].to(device, non_blocking=True)
            input_img_list = batch['input_img']
            input_img_list = list(map(lambda x: x.to(device, non_blocking=True), input_img_list))
            # New Feature
            if need_bg:
                bg_img = batch['bg_img'].to(device, non_blocking=True)
            # edited_img = batch['edited_img'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
            input_mode = batch['mode'].to(device, non_blocking=True)
            save_path = batch["path"][0]

            # process text ids to embeddings
            with torch.no_grad():
                input_txt_embs = t5_model.model(
                    input_ids=input_ids,
                    attention_mask=input_ids_attn_mask,
                )['last_hidden_state'].detach() # (B, length, channels)
            
            # process image ids to embeddings
            with torch.no_grad():
                _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                # bg_img_indices = bg_img_indices.reshape(bg_img.shape[0], -1) # type: ignore # (B, token_length)
                # _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                # input_img_indices = input_img_indices.reshape(input_img.shape[0], -1) # (B, token_length)
                ref_index_mask = []
                input_img_indices_lst = []
                for i, input_img in enumerate(input_img_list): # type: ignore
                    _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                    
                    input_img_indices_lst.append(input_img_indices.reshape(input_img.shape[0], -1))
                    ref_index_mask.append((torch.ones_like(input_img_indices_lst[-1], dtype=torch.int) * i).to(input_img_indices_lst[-1]))\
                
                # Add Background
                # _, _, [_, _, bg_img_indices] = vq_model.encode(bg_img) # type: ignore
                input_img_indices_lst.append(bg_img_indices.reshape(input_img.shape[0], -1)) # type: ignore
                ref_index_mask.append((torch.ones_like(input_img_indices_lst[-1], dtype=torch.int) * i).to(input_img_indices_lst[-1])) # type: ignore
                input_img_indices = torch.concat(input_img_indices_lst, dim=1)
                input_img_mask = torch.concat(ref_index_mask, dim=1)

                    # input_img_indices = torch.concat([input_img_indices, bg_img_indices], dim=1)
            
            # pdb.set_trace()
            B = input_img_mask.shape[0]
            qzshape = [B, args.codebook_embed_dim, latent_size, latent_size]
            t1 = time.time()
            index_sample = generate(
                gpt_model, input_txt_embs, input_img_indices, input_mode, latent_size ** 2, 
                emb_masks=None, 
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                # New Feature: Add ref index mask and activate special mask.
                mask_mode="ICBP",
                block_len=input_img_indices_lst[0].shape[1],
                num_ref=len(input_img_indices_lst),
                input_img_mask=input_img_mask,
                # device=device
                )
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
            t2 = time.time()
            samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            assert samples.shape[0] == 1
            decoder_time = time.time() - t2
            print(f"decoder takes about {decoder_time:.2f} seconds.")
            final_save_path = f"{save_path}/{timestep}_sample_cfg_{args.cfg_scale}_multi_cond_test.png"
            save_image(samples, final_save_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"image is saved to {final_save_path}")
        return f"{timestep}_sample_cfg_{args.cfg_scale}_multi_cond_test.png"
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-mode", type=str, choices=['dinov2', 'clip', 'clipseg'], default=None)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='your_path/ckpt/llamagen_t2i/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default="your_path/ckpt/CVPR2025_EditAR_release/editar_release/editar_release.pt", help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=['img_cls_emb', 'joint_cls_emb'], default='joint_cls_emb')
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input") # Text Prompt Token length
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--t5-path", type=str, default='your_path/ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--testset", type=str, choices=['custom'], default='custom')
    parser.add_argument("--dataset", type=str, default="your_path/result6")

    # New Feature
    parser.add_argument("--add_ref_embed", action="store_true", default=False)
    parser.add_argument("--multi-cond", action='store_true', default=False)
    parser.add_argument("--special-mask", action="store_true", default=False)
    parser.add_argument("--max_ref_num", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--additional-info", type=str, default=None)
    parser.add_argument("--concat-target", action="store_true", default=False)
    parser.add_argument("--test-version", action="store_true", default=False)
    parser.add_argument("--log-dir", type=str, default="your_path/code/EditAR/checkpoints/eval")

    # New Feature(8.26): Add Instruction Token to model
    parser.add_argument("--instruct-token-mode", type=str, choices=["casual", "special"], default=None)
    parser.add_argument("--instruct-token-num", type=int, default=120, help="The Instruction Token Number in model.")
    args = parser.parse_args()

    
    main(args)
