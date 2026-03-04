import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
                'dataset': 'custom',
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_edit_txt': edit_txt,
                }

def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="ssim_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,tgt_mask)


class PIE_Bench_Dataset(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args

        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path, 'mapping_file.json'), 'r') as file:
            self.dataset = json.load(file)

        self.mapping = {}
        for idx, key in enumerate(self.dataset.keys()):
            self.mapping[idx] = key
        
        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer


    def __len__(self,):
        return len(self.dataset)

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

    def mask_decode(self, encoded_mask, image_shape=[512,512]):
        length=image_shape[0]*image_shape[1]
        mask_array=np.zeros((length,))
        
        for i in range(0,len(encoded_mask),2):
            splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i]+j]=1
                
        mask_array=mask_array.reshape(image_shape[0], image_shape[1])
        # to avoid annotation errors in boundary
        mask_array[0,:]=1
        mask_array[-1,:]=1
        mask_array[:,0]=1
        mask_array[:,-1]=1
                
        return mask_array

    def __getitem__(self, index):

        # ['image_path', 'original_prompt', 'editing_prompt', 'editing_instruction', 'editing_type_id', 'blended_word', 'mask']
        
        data_path = self.dataset[self.mapping[index]]['image_path']

        input_img = Image.open(os.path.join(self.dataset_path, 'annotation_images', data_path)).convert('RGB')
        _input_img = input_img
        _mask = self.mask_decode(self.dataset[self.mapping[index]]["mask"])
        _mask = _mask[:,:,np.newaxis].repeat([3],axis=2)
        _original_prompt = self.dataset[self.mapping[index]]["original_prompt"].replace("[", "").replace("]", "")
        _editing_prompt = self.dataset[self.mapping[index]]["editing_prompt"].replace("[", "").replace("]", "")

        edit_txt = self.dataset[self.mapping[index]]['editing_instruction']

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

        save_input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        save_input_img = Image.fromarray(np.array(save_input_img).astype(np.uint8))
        save_input_img.save(f"{self.args.gpt_ckpt[:-3]}/PIE-bench/input/PIE_{index:08d}_input.png")
        # Create the content to be saved
        content = f"Edited Text:\n{edit_txt}"
        # Save the content to the file
        with open(f'{self.args.gpt_ckpt[:-3]}/PIE-bench/text/PIE_{index:08d}_txt.txt', "w") as file:
            file.write(content)

        return {
                'index': index,
                'image_path': data_path,
                'dataset': 'PIE',
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_original_prompt': _original_prompt,
                '_editing_prompt': _editing_prompt,
                '_mask': np.array(_mask),
                '_edit_txt': edit_txt,
                }


def main(args):
    random.seed(args.seed)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    latent_size = args.image_size // args.downsample_size
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
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
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

    save_path = args.gpt_ckpt[:-3]
    if args.testset == 'depth':
        MultiGen_Depth_Dataset_path = './data/'
        dataset = MultiGen_Depth_Eval_Dataset(
            args=args,
            dataset_path=MultiGen_Depth_Dataset_path,
            llm_tokenizer=t5_model.tokenizer,
            mode='train')

        from transformers import DPTImageProcessor, DPTForDepthEstimation
        depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        os.makedirs(save_path + '/depth/input', exist_ok=True)
        os.makedirs(save_path + '/depth/text', exist_ok=True)
        os.makedirs(save_path + '/depth/edit', exist_ok=True)
        os.makedirs(save_path + f'/depth/visualization/txt_{args.cfg_scale}/', exist_ok=True)   
        os.makedirs(save_path + f'/depth/samples/txt_{args.cfg_scale}/', exist_ok=True)   

    elif args.testset == 'canny':
        MultiGen_Canny_Dataset_path = './data/'
        dataset = MultiGen_Canny_Eval_Dataset(
            args=args,
            dataset_path=MultiGen_Canny_Dataset_path,
            llm_tokenizer=t5_model.tokenizer,
            mode='train')

        from kornia.filters import canny

        from torchmetrics.classification import BinaryF1Score
        from torchmetrics.image.fid import FrechetInceptionDistance as FID
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
        ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        f1 = BinaryF1Score().to(device)
        fid = FID(normalize=True).to(device)

        os.makedirs(save_path + '/canny/input', exist_ok=True)
        os.makedirs(save_path + '/canny/text', exist_ok=True)
        os.makedirs(save_path + '/canny/edit', exist_ok=True)
        os.makedirs(save_path + f'/canny/visualization/txt_{args.cfg_scale}/', exist_ok=True)   
        os.makedirs(save_path + f'/canny/samples/txt_{args.cfg_scale}/', exist_ok=True)   

    elif args.testset == 'conditionsegmentation':
        Condition_Segmentation_Dataset_path= './data/'
        dataset = Condition_Segmentation_Eval_Dataset(
            args=args,
            dataset_path=Condition_Segmentation_Dataset_path,
            llm_tokenizer=t5_model.tokenizer,
            mode='train')

        os.makedirs(save_path + '/conditionsegmentation/input', exist_ok=True)
        os.makedirs(save_path + '/conditionsegmentation/text', exist_ok=True)
        os.makedirs(save_path + '/conditionsegmentation/edit', exist_ok=True)   
        os.makedirs(save_path + '/conditionsegmentation/label', exist_ok=True)   
        os.makedirs(save_path + f'/conditionsegmentation/visualization/txt_{args.cfg_scale}/', exist_ok=True)   
        os.makedirs(save_path + f'/conditionsegmentation/samples/txt_{args.cfg_scale}/', exist_ok=True)   

    elif args.testset == 'PIE-bench':
        dataset = PIE_Bench_Dataset(args, dataset_path='./data/PIE_Bench_Dataset', llm_tokenizer=t5_model.tokenizer, mode='val')
        evaluation_result = []
        metrics = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
                   "ssim_unedit_part", "clip_similarity_source_image",
                   "clip_similarity_target_image", "clip_similarity_target_image_edit_part",
                   ]
        metrics_calculator = MetricsCalculator(device)

        os.makedirs(save_path + '/PIE-bench/input', exist_ok=True)
        os.makedirs(save_path + '/PIE-bench/text', exist_ok=True)
        os.makedirs(save_path + '/PIE-bench/edit', exist_ok=True)
        os.makedirs(save_path + f'/PIE-bench/visualization/txt_{args.cfg_scale}/', exist_ok=True)
        os.makedirs(save_path + f'/PIE-bench/samples/txt_{args.cfg_scale}/', exist_ok=True)

    elif args.testset == 'custom':
        dataset = Image_Folder_Dataset(args, dataset_path='eval_dataset/eval', llm_tokenizer=t5_model.tokenizer, mode='val')
        os.makedirs(save_path + '/custom/input', exist_ok=True)
        os.makedirs(save_path + '/custom/text', exist_ok=True)
        os.makedirs(save_path + '/custom/edit', exist_ok=True)
        os.makedirs(save_path + f'/custom/visualization/txt_{args.cfg_scale}/', exist_ok=True)
        os.makedirs(save_path + f'/custom/samples/txt_{args.cfg_scale}/', exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )


    for idx, batch in enumerate(loader):

        # input_img, edited_img, target_ids, input_ids, prompts
        input_img = batch['input_img'].to(device, non_blocking=True)
        # edited_img = batch['edited_img'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
        input_mode = batch['mode'].to(device, non_blocking=True)

        # if args.testset == 'conditionsegmentation':
        #     if os.path.exists(f"{save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['index'][0]:05d}.png"):
        #         print(f"Skip {save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['index'][0]:05d}.png")
        #         continue
        # else:
        #     if os.path.exists(f"{save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['dataset'][0]}_{batch['index'][0]:05d}_sample_txt_{args.cfg_scale}.png"):
        #         print(f"Skip {save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['dataset'][0]}_{batch['index'][0]:05d}_sample_txt_{args.cfg_scale}.png")
        #         continue

        # process text ids to embeddings
        with torch.no_grad():
            input_txt_embs = t5_model.model(
                input_ids=input_ids,
                attention_mask=input_ids_attn_mask,
            )['last_hidden_state'].detach()
        
        # process image ids to embeddings
        with torch.no_grad():
            _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
            input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)

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

        if args.testset == 'PIE-bench':
            mask = batch['_mask'][0].cpu().numpy() # (512, 512, 3) [0, 255]
            src_image = batch['_input_img'][0].cpu().numpy()
            src_image = Image.fromarray(np.uint8(src_image))
            samples = torch.clamp(samples, -1, 1)
            tgt_image = (samples[0].cpu().permute(1,2,0)+1)/2 * 255.
            tgt_image = Image.fromarray(np.uint8(tgt_image))
            src_prompt = batch['_original_prompt'][0]
            tgt_prompt = batch['_editing_prompt'][0]

            evaluation_result = [batch['index'][0].item(), batch['image_path'][0]]
            for metric in metrics:
                cal_metric = calculate_metric(metrics_calculator,
                                                metric,
                                                src_image=src_image,
                                                tgt_image=tgt_image,
                                                src_mask=mask,
                                                tgt_mask=mask, src_prompt=src_prompt, tgt_prompt=tgt_prompt)
                
                print(metric, cal_metric)
                evaluation_result.append(cal_metric)

            with open(save_path+f'/{args.testset}/samples/txt_{args.cfg_scale}/PIE_Bench.csv','a+',newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)

        if args.testset == 'conditionsegmentation':
            samples = torch.clamp(samples, -1, 1)
            pass

        if args.testset == 'depth':
            label = batch['_input_img'][0]
            label = np.array(label)
            label = (label - label.min()) / (label.max() - label.min())
            label = Image.fromarray(np.array(label * 255.).astype(np.uint8)).convert('L')
            label = torch.from_numpy(np.array(label))
            samples = torch.clamp(samples, -1, 1)
            depth_input_images = (samples[0].cpu().permute(1,2,0)+1)/2 * 255.
            depth_input_images = Image.fromarray(np.uint8(np.array(depth_input_images)))
            depth_model_input = depth_processor(images=[depth_input_images], return_tensors="pt")
            with torch.no_grad():
                outputs = depth_model(**depth_model_input)
                predicted_depth = outputs.predicted_depth

            predicted_depth = predicted_depth[0]
            predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            predicted_depth = Image.fromarray(np.array(predicted_depth * 255.).astype(np.uint8))
            predicted_depth = predicted_depth.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
            pred = predicted_depth.convert('L')
            pred = torch.from_numpy(np.array(pred))
            per_pixel_mse = torch.sqrt(F.mse_loss(pred.float(), label.float()))
            print('mse: ', per_pixel_mse.cpu().item())

            evaluation_result = [batch['index'][0].cpu().item()]
            evaluation_result.append(per_pixel_mse.cpu().item())

            with open(save_path+f'/{args.testset}/samples/txt_{args.cfg_scale}/depth.csv','a+',newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)

        if args.testset == 'canny':
            samples = torch.clamp(samples, -1, 1)
            pred_condition = (samples+1)/2

            # process input image for the canny edge image
            low_threshold = 0.1
            high_threshold = 0.2
            _, pred_condition = canny(pred_condition, low_threshold, high_threshold)
            pred_condition = (pred_condition.cpu() * 255.).to(torch.uint8)
            gt_condition = (batch['_input_img'] * 255.).to(torch.uint8)
            
            # Assuming ssim and psnr instances are created and moved to the correct device beforehand
            ssim_score = ssim((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))
            psnr_score = psnr((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))

            gt_condition[gt_condition == 255] = 1
            pred_condition[pred_condition == 255] = 1

            f1_score = f1(pred_condition.flatten(), gt_condition.flatten())
            print( 'f1: ', f1_score.item(),  ' psnr: ', psnr_score.item(), ' ssim: ', ssim_score.item())

            evaluation_result = [batch['index'][0].item()]
            evaluation_result.append(f1_score.cpu().item())
            evaluation_result.append(psnr_score.cpu().item())
            evaluation_result.append(ssim_score.cpu().item())

            with open(save_path+f'/{args.testset}/samples/txt_{args.cfg_scale}/canny.csv','a+',newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)

        if args.testset == 'conditionsegmentation':
            save_image(samples, f"{save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['index'][0]:05d}.png", nrow=4, normalize=True, value_range=(-1, 1))
        else:
            save_image(samples, f"{save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['dataset'][0]}_{batch['index'][0]:05d}_sample_txt_{args.cfg_scale}.png", nrow=4, normalize=True, value_range=(-1, 1))
            print(f"image is saved to {save_path}/{args.testset}/samples/txt_{args.cfg_scale}/{batch['dataset'][0]}_{batch['index'][0]:05d}_sample_txt_{args.cfg_scale}.png")


        vis_input_img = batch['input_img'][0]
        vis_input_img = (vis_input_img.permute(1,2,0) + 1) / 2.
        vis_edited_img = batch['edited_img'][0]
        vis_edited_img = (vis_edited_img.permute(1,2,0) + 1) / 2.
        vis_text = batch['_edit_txt'][0]
        vis_sample_img = (samples[0].cpu().permute(1, 2, 0) + 1) / 2.
        vis_img = torch.cat([vis_input_img, vis_edited_img, vis_sample_img], dim=1)

        plt.imshow(vis_img)
        plt.title(vis_text, fontsize=8)
        plt.savefig(f"{save_path}/{args.testset}/visualization/txt_{args.cfg_scale}/{batch['index'][0]:05d}.png", dpi=512)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-mode", type=str, choices=['dinov2', 'clip', 'clipseg'], default=None)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='./pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=['img_cls_emb', 'joint_cls_emb'], default='joint_cls_emb')
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
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
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--testset", type=str, choices=['custom', 'PIE-bench', 'depth', 'canny', 'conditionsegmentation'], default='train')
    args = parser.parse_args()
    main(args)
