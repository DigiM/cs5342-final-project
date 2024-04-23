import torch

from xclip.clip import clip

def generate_categories(data):
    text_aug = f"{{}}"
    categories = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.categories])

    return categories