import torch
import torchvision
from torch import nn
from s2igan.utils import set_non_grad


class ImageEncoder(nn.Module):
    """
        image encoder pretrained on imagenet
    """

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        weights = torchvision.models.get_weight("Inception_V3_Weights.DEFAULT")
        model = torchvision.models.inception_v3(weights=weights)
        model.AuxLogits = None
        model.aux_logits = False
        set_non_grad(model)
        self.model = model
        self.model.fc = nn.Linear(2048, output_dim)

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, img):
        """
            img: (-1, 3, 299, 299)
            out: (-1, output_dim)
        """
        return self.model(img)
