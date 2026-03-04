from PIL import Image
import shutil
import os
import numpy as np

# DINO-Score
from typing import Union, List
from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel
import ipdb

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

import sys
sys.path.append("your_path/code/L2h_code")
from agent.BEN2 import BEN_Base
from PIL import Image
import numpy as np
from typing import List, Any
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from tqdm import tqdm
device="cuda:0"
BNE_model = BEN_Base().to(device).eval()
BNE_model.loadcheckpoints('your_path/ckpt/BEN2/BEN2_Base.pth')

model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)


def convert_to_rgb(img_path, model):
    img_tensor = np.array(Image.open(img_path))
    def segment(image_for_bg):
        image_for_bg = Image.fromarray(image_for_bg).convert("RGB")
        foreground = model.inference(image_for_bg, refine_foreground=True)
        # 生成背景图
        # 将PIL图像转换为numpy数组
        image_np = np.array(image_for_bg)
        foreground_np = np.array(foreground)
        background_np = np.where(foreground_np[:, :, 3] > 0, True, False)[:, :, np.newaxis]
        background_np = np.repeat(background_np, 3, 2)
        image_np[background_np] = 0
        return image_np, np.array(foreground)
    bg, fore = segment(img_tensor)
    return Image.fromarray(img_tensor), Image.fromarray(bg), Image.fromarray(fore) 

@torch.no_grad()
def dino_score(
    images: Union[Image.Image, List[Image.Image]],
    add_images: Union[Image.Image, List[Image.Image]],    
) -> Tensor:


    if not isinstance(images, list):
        images = [images]
    else:  # unwrap into list
        images = list(images)

    if not isinstance(add_images, list):
        add_images = [add_images]
    else:  # unwrap into list
        add_images = list(add_images)

    images = [transform(i).to(device) for i in images]
    add_images = [transform(i).to(device) for i in add_images]

    scores = []
    for img, add_img in zip(images, add_images):
        inputs = torch.stack([img, add_img])
        # Get DINO features
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state  # ViT backbone features
        emb_img1, emb_img2 = last_hidden_states[0, 0], last_hidden_states[1, 0]  # Get cls token (0-th token) for each img
        scores.append(F.cosine_similarity(emb_img1, emb_img2, dim=0))
    scores = torch.stack(scores, dim=0)
    return scores.mean()*100

def filter(path, delta:float=20):
    foreground_img_path = os.path.join(path, "real.png")
    real_img = Image.open(foreground_img_path)
    def masked(image):
        image = np.array(image)
        mask = image[:, :, 3] < image[:, :, 3].mean()
        image = image[:, :, :3]
        image[mask]=0
        img=Image.fromarray(image)
        return img
    ref0_path = os.path.join(path, "ref", "ref_0.png")
    ref1_path = os.path.join(path, "ref", "ref_1.png")
    _, __, fore_0 = convert_to_rgb(ref0_path, BNE_model)
    _, __, fore_1 = convert_to_rgb(ref1_path, BNE_model)
    final_score = min(dino_score(real_img, masked(fore_0)), dino_score(real_img, masked(fore_1)))
    print(f"Score: {final_score}")
    if final_score > delta:
        pass
    else:
        shutil.rmtree(path)
        print(f"Delete : {path}")
    return final_score

def filter2(path, delta:float=20):
    foreground_img_path = os.path.join(path, "real.png")
    real_img = Image.open(foreground_img_path)
    def masked(image):
        image = np.array(image)
        mask = image[:, :, 3] < image[:, :, 3].mean()
        image = image[:, :, :3]
        image[mask]=0
        img=Image.fromarray(image)
        return img
    ref0_path = os.path.join(path, "ref", "ref_0.png")
    ref1_path = os.path.join(path, "ref", "ref_1.png")
    _, __, fore_0 = convert_to_rgb(ref0_path, BNE_model)
    _, __, fore_1 = convert_to_rgb(ref1_path, BNE_model)
    final_score = min(dino_score(real_img, masked(fore_0)), dino_score(real_img, masked(fore_1)))
    print(f"Score: {final_score}")
    if final_score > delta:
        pass
    else:
        shutil.rmtree(path)
        print(f"Delete : {path}")
    return final_score

def main():
    import random
    dp = "your_path/data/dreamRelationGenerateds2"
    record_lst = os.listdir(dp)
    record_lst = random.sample(record_lst, 1000)
    scores = []
    for i, record_name in tqdm(enumerate(record_lst), total=len(record_lst)):
        tp = os.path.join(dp, record_name)
        score = filter(tp)
        scores.append(score)

    torch.save(scores, "./score2.pt")

    print(f"Success")
if __name__ == '__main__':
    main()
    exit(0)
