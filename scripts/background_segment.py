import sys
from BEN2 import BEN_Base
from PIL import Image
import numpy as np
from typing import List, Any
import os
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
device="cuda:0"
model = BEN_Base().to(device).eval()
model.loadcheckpoints('your_path/ckpt/BEN2/BEN2_Base.pth')


def segment_and_save(img_path, save_path):
    global model
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
    # 执行前景分割
    img, bg, fore = convert_to_rgb(img_path, model)


    # 保存背景图
    bg.save(os.path.join(save_path, "background.png"))
    fore.save(os.path.join(save_path, "foreground.png"))
    print(f"Success Save in {save_path}")
    
    
def main():
    dp = "Your_dataset/"
    record_lst = os.listdir(dp)
    for i, record_name in tqdm(enumerate(record_lst), total=len(record_lst)):
        tmp_dp = os.path.join(dp, record_name)
        segment_and_save(os.path.join(tmp_dp, "real.png"), tmp_dp)

if __name__ == '__main__':
    main()
    print(f"Finished")
