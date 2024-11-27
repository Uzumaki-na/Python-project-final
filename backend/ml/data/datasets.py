import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_normalization_values(data_paths):
    """Calculate mean and std for normalization"""
    means = []
    stds = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print("Calculating dataset statistics...")
    for img_path in tqdm(data_paths):
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        means.append(torch.mean(image, dim=[1,2]))
        stds.append(torch.std(image, dim=[1,2]))
    
    mean = torch.mean(torch.stack(means), dim=0)
    std = torch.mean(torch.stack(stds), dim=0)
    return mean, std

def get_transform():
    """
    Get the standard transform pipeline used in training.
    This should match the transforms used in training exactly.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet-B0 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats since we used them in training
    ])

class SkinCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None, calculate_stats=False):
        """
        Args:
            data_dir (string): Directory with all the images and metadata.csv
            transform (callable, optional): Optional transform to be applied on a sample.
            calculate_stats (bool): Whether to calculate normalization statistics
        """
        self.data_dir = data_dir
        
        # Read metadata
        self.metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        if calculate_stats:
            # Get all image paths
            image_paths = [os.path.join(data_dir, 'images', img_id + '.jpg') 
                         for img_id in self.metadata['image_id']]
            mean, std = calculate_normalization_values(image_paths)
        else:
            # Use pre-calculated values (you'll need to calculate these once)
            mean = torch.tensor([0.7, 0.7, 0.7])  # placeholder values
            std = torch.tensor([0.15, 0.15, 0.15])  # placeholder values
        
        self.transform = transform if transform else get_transform()
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.data_dir, 'images', self.metadata.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(1 if self.metadata.iloc[idx]['target'] == 'malignant' else 0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class MalariaDataset(Dataset):
    def __init__(self, data_dir, transform=None, calculate_stats=False):
        """
        Args:
            data_dir (string): Directory with all the images organized in subdirectories
            transform (callable, optional): Optional transform to be applied on a sample.
            calculate_stats (bool): Whether to calculate normalization statistics
        """
        self.data_dir = data_dir
        
        # Get all image paths and labels
        self.data = []
        for class_name in ['Parasitized', 'Uninfected']:
            class_dir = os.path.join(data_dir, class_name)
            class_label = 0 if class_name == 'Parasitized' else 1  # Fixed: 0 for Parasitized, 1 for Uninfected
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    self.data.append((os.path.join(class_dir, img_name), class_label))
        
        if calculate_stats:
            # Get all image paths
            image_paths = [path for path, _ in self.data]
            mean, std = calculate_normalization_values(image_paths)
        else:
            # Use pre-calculated values
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.25, 0.25, 0.25])
        
        self.transform = transform if transform else get_transform()
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def preprocess_image(image, model_type='malaria'):
    """
    Preprocess a single image for inference.
    Args:
        image: Either a file path or a PIL Image object
        model_type: Either 'malaria' or 'skin_cancer'
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model inference
    """
    try:
        # Convert to RGB if needed
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Model-specific normalization values
        if model_type == 'skin_cancer':
            mean = [0.7012, 0.5517, 0.4875]
            std = [0.1420, 0.1520, 0.1696]
        else:  # malaria
            mean = [0.5, 0.5, 0.5]
            std = [0.25, 0.25, 0.25]
            
        # Get the transform pipeline with model-specific normalization
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet-B0 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Apply transforms and add batch dimension
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

class SafeImageDataset(Dataset):
    """Base dataset class with error handling"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else get_transform()
        
    def __getitem__(self, idx):
        try:
            image_path, label = self.get_item_data(idx)
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # Return a default item or raise the error
            raise e
            
    def get_item_data(self, idx):
        """To be implemented by child classes"""
        raise NotImplementedError
