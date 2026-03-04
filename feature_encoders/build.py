from transformers import AutoImageProcessor, Dinov2Model
from transformers import AutoProcessor, CLIPSegVisionModel, CLIPVisionModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np

# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]
# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# image = np.random.uniform(0, 1, (8, 512, 512, 3)).astype(np.float32)
# image = torch.tensor(image)
# inputs = image_processor(image, return_tensors="pt")

class Semantic_Encoder(torch.nn.Module):

    def __init__(self, mode, precision, device):
        super().__init__()
        self.mode = mode
        self.precision = precision
        self.device = device
        if self.mode == 'dinov2':
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.precision).to(self.device)
            print('load dinov2 for distillation')
        if self.mode == 'clip':
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(self.precision).to(self.device)
            print('load clip for distillation')
        if self.mode == 'clipseg':
            self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.precision).to(self.device)
            print('load clipseg for distillation')

    def compute_distill_loss(self, edited_img, llm_features):

        if self.mode == 'dinov2':
            image = (edited_img.permute(0, 2, 3, 1).contiguous() + 1) / 2
            inputs = self.image_processor(image, return_tensors="pt", do_rescale=False)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.precision).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        
        if self.mode == 'clip' or self.mode == 'clipseg':
            image = (edited_img.permute(0, 2, 3, 1).contiguous() + 1) / 2
            inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.precision).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        
        distill_loss = F.mse_loss(llm_features, last_hidden_states[:, 1:].detach())

        return distill_loss
