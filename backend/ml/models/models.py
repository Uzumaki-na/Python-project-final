import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(1280, num_classes)
        
        # Register buffers for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        
    def preprocess(self, x):
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        
    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)
        
    def load_state_dict(self, state_dict, strict=True):
        """Custom state dict loading to handle model architecture changes"""
        try:
            # First try loading with strict=True
            return super().load_state_dict(state_dict, strict=strict)
        except Exception as e:
            if not strict:
                # If strict=False, try to load what we can
                model_state_dict = self.state_dict()
                for name, param in state_dict.items():
                    if name in model_state_dict:
                        if isinstance(param, torch.nn.Parameter):
                            param = param.data
                        try:
                            model_state_dict[name].copy_(param)
                        except Exception:
                            print(f'Failed to copy parameter {name}')
                return model_state_dict
            raise e

class SkinCancerModel(EfficientNetModel):
    def __init__(self):
        super().__init__(num_classes=2)  # 2 classes: benign/malignant
        # Override with skin-specific normalization
        self.register_buffer('mean', torch.tensor([0.7012, 0.5517, 0.4875]))
        self.register_buffer('std', torch.tensor([0.1420, 0.1520, 0.1696]))

class MalariaModel(EfficientNetModel):
    def __init__(self):
        super().__init__(num_classes=2)  # 2 classes: Parasitized/Uninfected
        # Override with microscopy-specific normalization
        self.register_buffer('mean', torch.tensor([0.5302, 0.5302, 0.5302]))
        self.register_buffer('std', torch.tensor([0.1967, 0.1967, 0.1967]))
