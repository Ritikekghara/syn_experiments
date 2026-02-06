import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2):
    model = models.resnet50(pretrained=True)

    # Freeze early layers, unfreeze layer4 (last convolutional block)
    for param in model.parameters():
        param.requires_grad = False
    
    # # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Checkpoint was trained with Sequential containing only Linear layer (no Dropout)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model
