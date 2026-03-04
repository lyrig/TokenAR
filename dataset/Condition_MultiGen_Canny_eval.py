import pdb

from datasets import load_from_disk, concatenate_datasets
import io
import numpy as np
import PIL
from PIL import Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

from kornia.filters import canny

# MultiGen dataset
class MultiGen_Canny_Eval_Dataset(Dataset):
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        # MultiGen Dataset path
        self.dataset_path = load_from_disk(dataset_path+'/MultiGen-20M_depth_eval_HF')

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        # self.mask_generator = mask_generator()


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
        # IMG_START = self.llm_tokenizer.vocab.boi_id
        # IMG_END = self.llm_tokenizer.vocab.eoi_id
        # Loading Path...
        data = self.dataset_path[index]

        # convert into torch style
        input_img = data['image']
        edited_img = data['image']
        input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
        edited_img = Image.open(io.BytesIO(edited_img['bytes'])).convert('RGB')

        # input_txt = data['original_prompt']
        # edited_txt = data['edited_prompt']
        edit_txt = 'Given the canny edge image, generate an image following the description: '+data['text']

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

        # process input image for the canny edge image
        input_img = (input_img+1) / 2
        low_threshold = 0.1
        high_threshold = 0.2
        _, input_img = canny(input_img[None], low_threshold, high_threshold)
        _input_img = input_img[0]
        input_img = input_img.repeat(1, 3, 1, 1)[0]
        input_img = input_img * 2 -1

        save_input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        save_input_img = Image.fromarray(np.array(save_input_img).astype(np.uint8))
        save_input_img.save(f"{self.args.gpt_ckpt[:-3]}/canny/input/canny_eval_{index:05d}_input.png")
        save_edited_img = (edited_img.permute(1,2,0)+1)/2 * 255.
        save_edited_img = Image.fromarray(np.array(save_edited_img).astype(np.uint8))
        save_edited_img.save(f"{self.args.gpt_ckpt[:-3]}/canny/edit/canny_eval_{index:05d}_edit.png")
        # Create the content to be saved
        content = f"Edited Text:\n{edit_txt}"
        # Save the content to the file
        with open(f'{self.args.gpt_ckpt[:-3]}/canny/text/canny_eval_{index:05d}_txt.txt', "w") as file:
            file.write(content)

        return {
                'index': index,
                'dataset': 'canny_eval',
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                # 'target_ids': target_ids,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': _input_img,
                '_edit_txt': edit_txt,
                # '_input_ids': _input_ids,
                # '_input_ids_attn_mask': _input_ids_attn_mask,
                # '_target_ids': _target_ids,
                }

