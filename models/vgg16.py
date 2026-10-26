from torchvision import models
from config import DEVICE


def load_vgg16():
    model = models.vgg16(pretrained=True)
    model = model.eval().to(DEVICE)
    return model
