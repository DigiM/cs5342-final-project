import torch
import torch.nn as nn

from .prompt_learner import TextEncoder, PromptLearner

class CategoryPredictor(nn.Module):
    def __init__(self, clip_model, categories):
        super().__init__()
        
        self.prompt_learner = PromptLearner(categories, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits