from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch
import os
import shutil
from tqdm import tqdm
def filter(device, record_lst, data_path):
    model_id = "your_path/ckpt/gemma-3n-E4B-it"

    model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    for i, record_name in tqdm(enumerate(record_lst), total=len(record_lst)):
        path = os.path.join(data_path, record_name)
        try:
            with open(os.path.join(path, "description.txt"), "r") as f:
                description = f.read()
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.join(path, "real.png")},
                        {"type": "image", "image": os.path.join(path, "ref", "ref_0.png")},
                        {"type": "image", "image": os.path.join(path, "ref", "ref_1.png")},
                        {"type": "text", "text": f"Is the last two image contains in the first image? And the Picture Color is normal. Not too extreme. And the image fit the description: {description}. Reply Yes or No."}
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            if "No" in decoded:
                print(f"{decoded}, {path}")
                shutil.rmtree(path) # Delete the record
            else:
                pass
        except Exception as e:
            print(f"出现错误; {e}")
            shutil.rmtree(path) # Delete the record
if __name__ == "__main__":
    dp = "your_path/data/dreamRelationGenerateds3"
    record_lst = os.listdir(dp)
    print(f"{len(record_lst)}")
    filter("cuda:0", record_lst, dp)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.
