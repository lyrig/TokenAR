import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import random
def get_all_png_paths_glob(directory_path: str, return_format:str="str") -> list[Path]|list[str]:
    """
    使用 Path.glob() 获取指定目录下及其子目录中所有 PNG 图片的 Path 对象。

    Args:
        directory_path: 要搜索的根目录路径（字符串形式）。

    Returns:
        一个包含所有找到的 PNG 图片 Path 对象的列表。
    """
    base_path = Path(directory_path)
    # 使用 '**/*.png' 模式递归查找所有 .png 文件
    png_files = list(base_path.glob('**/*.png')) + list(base_path.glob('**/*.webp'))
    if return_format == "str":
        return list(map(lambda x: str(x), png_files))
    elif return_format == "Path":
        return png_files
    else:
        raise f"return_format is invalid, expected `str` or `Path`, but get {return_format}" # type: ignore

# New Feature; Generated through the Dataset
class MultiCond_IP_Plus_Dataset(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 need_bg:bool=False,
                 max_cond:int=4,
                 world_size:int=1,
                 rank:int=0
                 ):

        self.args = args
        self.mode = mode
        # Dataset path
        # self.dataset_path
        try:
            self.dataset_path = list(map(lambda x: os.path.join(dataset_path, x), list(sorted(os.listdir(dataset_path), key=lambda x: int(x.split('_')[-1])))))
        except Exception as e:
            print(f"Sort Error: {e}")
            self.dataset_path = list(map(lambda x: os.path.join(dataset_path, x), list(sorted(os.listdir(dataset_path)))))
        self.dataset_path = self.dataset_path[:1000] # 记得删掉
        self.dataset_path = list(filter(lambda x: len(get_all_png_paths_glob(os.path.join(x, "ref"))) != 0 and os.path.exists(os.path.join(x, "real.png")), self.dataset_path))[rank::world_size]

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer

        # New Feature: Multiple condition(with bg)
        self.return_bg = need_bg
        self.max_ref = max_cond - 1 if need_bg else max_cond

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

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512, target:bool=False) -> torch.Tensor:
        """
        将PIL图像转换为用于VQGAN的PyTorch张量，并应用数据增强。

        Args:
            img (PIL.Image): 输入的PIL图像。
            target_image_size (int): 目标图像尺寸，例如 512。

        Returns:
            torch.Tensor: 预处理后的图像张量，带有一个批次维度。
        """
        img = img.resize((target_image_size, target_image_size))
        if not target and self.mode == 'train':
            # 定义预处理和数据增强的变换
            # 实验发现：数据增强并不会使得模型的生成效果更好，尤其是在Ref图像较小的数据集中
            transform = transforms.Compose([
                
                # 2. 随机平移和旋转（使用 RandomAffine）
                # 这里 RandomAffine 结合了旋转和平移，可以替代 RandomRotation
                # transforms.RandomAffine(
                #     degrees=5,            # 随机旋转 -5 到 +5 度
                #     scale=(0.9, 1.1)      # 随机缩放 ±10%
                # ),
                # 4. 随机水平/垂直翻转
                # transforms.RandomHorizontalFlip(p=0.5),
                
                # transforms.RandomVerticalFlip(p=0.5), 
                
                # 5. 转换为张量
                transforms.ToTensor(),
                
                # 6. 归一化到 [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([

                # 5. 转换为张量
                transforms.ToTensor(),
                
                # 6. 归一化到 [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # 对图像应用所有变换
        tensor_img = transform(img)

        return tensor_img

    
    def _vqgan_input_from_old(self, img: PIL.Image, target_image_size=512) -> torch.Tensor: # type: ignore
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
        input_path_list = get_all_png_paths_glob(os.path.join(data_path, "ref"))
        edited_path = os.path.join(data_path, "real.png")
        input_img_list = [self._vqgan_input_from(self._whiten_transparency(Image.open(input_path).convert('RGB'))) for input_path in input_path_list]
        edited_img = Image.open(edited_path).convert("RGB")
        # New Feature: Add BG
        if self.return_bg:
            bg_path = os.path.join(data_path, "background.png")
            bg_img = Image.open(bg_path).convert("RGB")
        

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

        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img, target=True)

        if self.return_bg:
            bg_img = self._whiten_transparency(bg_img) # type: ignore
            bg_img = self._vqgan_input_from(bg_img, target=True)

        input_img_list = input_img_list[:self.max_ref]
        


        if len(input_img_list) < self.max_ref:
            input_img_list.extend([torch.zeros_like(input_img_list[0]).float() for i in range(self.max_ref - len(input_img_list))])
        
        
        if self.return_bg:
            if self.args.instruct_token_mode == "special":
                index = random.randint(0, len(input_img_list)-1) if self.mode == 'train' else len(input_img_list)-1
                if index == len(input_img_list)-1:
                    target = edited_img
                else:
                    target=input_img_list[index]
            else:
                index = 0
                target = edited_img
            return {
                    'index': index,
                    'dataset': os.path.basename(data_path),
                    'mode': 1,
                    # Text
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,

                    # Img
                    'input_img': input_img_list,
                    'bg_img': bg_img, # type: ignore
                    'edited_img': target, # Target Image

                    # Others
                    'path': data_path,

                    # Instruct Index
                    "instruct": index
                    }
        else:
            return {
                    'index': index,
                    'dataset': os.path.basename(data_path)[:-4],
                    'mode': 1,

                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img_list,
                    'edited_img': edited_img,

                    'path': data_path,
                    "instruct":0
                    }

if __name__ == "__main__":
    pass