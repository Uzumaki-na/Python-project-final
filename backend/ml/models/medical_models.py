import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Create attention masks
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class MedicalEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        # Load pretrained EfficientNet
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Remove original classifier
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Add medical-specific layers
        self.attention = SpatialAttention()
        self.dropout = nn.Dropout(dropout_rate)
        
        # New classifier with temperature scaling for confidence calibration
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Apply attention
        attn = self.attention(x)
        x = x * attn
        
        # Classification with temperature scaling
        x = self.classifier(x)
        
        if self.training:
            return x
        else:
            # Apply temperature scaling during inference
            return x / self.temperature

class SkinCancerModel(MedicalEfficientNet):
    def __init__(self):
        super().__init__(num_classes=2)  # benign/malignant
        
        # Skin-specific preprocessing
        self.register_buffer('mean', torch.tensor([0.7012, 0.5517, 0.4875]))  # Skin image stats
        self.register_buffer('std', torch.tensor([0.1420, 0.1520, 0.1696]))
        
    def preprocess(self, x):
        # Normalize using skin-specific statistics
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
    
    def forward(self, x):
        x = self.preprocess(x)
        return super().forward(x)

class MalariaModel(MedicalEfficientNet):
    def __init__(self):
        super().__init__(num_classes=2)  # Parasitized/Uninfected
        
        # Microscopy-specific preprocessing
        self.register_buffer('mean', torch.tensor([0.5302, 0.5302, 0.5302]))  # Microscopy stats
        self.register_buffer('std', torch.tensor([0.1967, 0.1967, 0.1967]))
        
    def preprocess(self, x):
        # Normalize using microscopy-specific statistics
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
    
    def forward(self, x):
        x = self.preprocess(x)
        return super().forward(x)

def get_confidence_score(logits):
    """Convert logits to calibrated confidence scores"""
    probs = F.softmax(logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    return confidence, prediction
