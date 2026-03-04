import torch
import torch.nn.functional as F

import pdb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
# New Feature; Generated through the Dataset
class MultiCond_IP_Dataset(Dataset):
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
        self.dataset_path = list(map(lambda x: os.path.join(dataset_path, x), list(sorted(os.listdir(dataset_path)))))
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
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB") # type: ignore

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor: # type: ignore
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS) # type: ignore

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
        edited_path = os.path.join(data_path, "real.png")
        input_img = Image.open(input_path).convert('RGB')
        edited_img = Image.open(edited_path).convert("RGB")
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
        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img)

        if self.return_bg:
            bg_img = self._whiten_transparency(bg_img) # type: ignore
            bg_img = self._vqgan_input_from(bg_img)

        if self.return_bg:
            return {
                    'index': index,
                    'dataset': os.path.basename(data_path),
                    'mode': 1,
                    # Text
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,

                    # Img
                    'input_img': input_img,
                    'bg_img': bg_img, # type: ignore
                    'edited_img': edited_img, # Target Image

                    # Others
                    # '_input_img': np.array(_input_img),
                    # '_bg_img': np.array(_bg_img), # type: ignore
                    # '_edit_txt': edit_txt,
                    'path': data_path
                    }
        else:
            return {
                    'index': index,
                    'dataset': os.path.basename(data_path)[:-4],
                    'mode': 1,

                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'edited_img': edited_img,

                    # '_input_img': np.array(_input_img),
                    # '_edit_txt': edit_txt,
                    'path': data_path
                    }
