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

from autoregressive.sample.metrics_calculator import MetricsCalculator

import csv

from dataset.Condition_MultiGen_Depth_eval import MultiGen_Depth_Eval_Dataset
from dataset.Condition_MultiGen_Canny_eval import MultiGen_Canny_Eval_Dataset
from dataset.Condition_Segmentation_eval import Condition_Segmentation_Eval_Dataset

from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel,
    UniPCMultistepScheduler, DDIMScheduler,
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline, AutoencoderKL
)

class Image_Folder_Dataset(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        # Dataset path
        self.dataset_path = natsorted(glob.glob(os.path.join(dataset_path, '*_input.png')))
        # self.dataset_path = natsorted(glob.glob(os.path.join(dataset_path, '*_input.jpg')))

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer


    def __len__(self,):
        return len(self.dataset_path)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def __getitem__(self, index):
        data_path = self.dataset_path[index]

        input_img = Image.open(data_path).convert('RGB')
        _input_img = input_img

        with open(data_path[:-10]+'_txt.txt', 'r') as file:
            edit_txt = file.read()

        edit_text_tokens_and_mask = self.llm_tokenizer(
            edit_txt,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        edit_txt_token = edit_text_tokens_and_mask['input_ids']
        edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']

        input_ids = edit_txt_token[0]
        input_ids_attn_mask = edit_txt_attn_mask[0]

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = - torch.ones(input_img.shape)

        return {
                'index': index,
                'dataset': os.path.basename(data_path)[:-4],
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_edit_txt': edit_txt,
                }

# New Feature; Generated through the Dataset
class Image_Folder_Dataset2(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 need_bg:bool=False
                 ):

        self.args = args
        # Dataset path
        # self.dataset_path
        raw_list = list(sorted(os.listdir(dataset_path), key=lambda x: int(x.split(".")[0].split("_")[1])))
        self.dataset_path = list(map(lambda x: os.path.join(dataset_path, x), raw_list))
        # self.dataset_path = natsorted(glob.glob(os.path.join(dataset_path, '*_input.jpg')))

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer

        # New Feature: Multiple condition(with bg)
        self.return_bg = need_bg


    def __len__(self,):
        return len(self.dataset_path)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image: # type: ignore
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def __getitem__(self, index):
        data_path = self.dataset_path[index]
        input_path = os.path.join(data_path, "ref", "ref.png")
        input_img = Image.open(input_path).convert('RGB')

        # New Feature: Add BG
        if self.return_bg:
            bg_path = os.path.join(data_path, "background.png")
            bg_img = Image.open(bg_path).convert("RGB")
            _bg_img = bg_img
        _input_img = input_img
        

        with open(os.path.join(data_path, "description.txt"), 'r') as file:
            edit_txt = file.read()

        edit_text_tokens_and_mask = self.llm_tokenizer(
            edit_txt,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        edit_txt_token = edit_text_tokens_and_mask['input_ids']
        edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']

        input_ids = edit_txt_token[0]
        input_ids_attn_mask = edit_txt_attn_mask[0]

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)

        if self.return_bg:
            bg_img = self._whiten_transparency(bg_img) # type: ignore
            bg_img = self._vqgan_input_from(bg_img)
        edited_img = - torch.ones(input_img.shape)
        if self.return_bg:
            return {
                    'index': index,
                    'dataset': data_path.split("/")[-2],
                    'mode': 1,
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'bg_img': bg_img,
                    'edited_img': edited_img,
                    '_input_img': np.array(_input_img),
                    '_bg_img': np.array(_bg_img),
                    '_edit_txt': edit_txt,
                    'path': data_path
                    }
        else:
            return {
                    'index': index,
                    'dataset': data_path.split("/")[-2],
                    'mode': 1,
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'edited_img': edited_img,
                    '_input_img': np.array(_input_img),
                    '_edit_txt': edit_txt,
                    'path': data_path
                    }


def main(args):
    random.seed(args.seed)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        multi_cond=need_bg
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
    data_path = "your_path/data/dataset/unzipSubject200K_test"

    dataset = Image_Folder_Dataset2(args, dataset_path=data_path, llm_tokenizer=t5_model.tokenizer, mode='val', need_bg=need_bg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )


    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating Processing:"):

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
    parser.add_argument("--multi-cond", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
