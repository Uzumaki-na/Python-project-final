import os
import time
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import timm
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Update model directory to the correct location
model_dir = Path("backend/ml/models/trained")
model_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Model directory: {model_dir}")

def create_data_loaders(data_dir, batch_size=128, is_skin_cancer=False):
    """Optimized data loading with model-specific settings"""
    logger.info(f"Creating data loaders for {data_dir}")
    
    # Enhanced transforms for skin cancer model
    if is_skin_cancer:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Larger size for skin cancer
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Medical images can be flipped
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        sample_ratio = 0.5  # Use more data for skin cancer
    else:
        train_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        sample_ratio = 0.2
    
    try:
        full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        full_val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
        
        # Get class distribution
        class_counts = {}
        for _, label in full_train_dataset:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
            
        # Calculate class weights for balanced sampling
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        
        # Create sample weights for balanced sampling
        sample_weights = [class_weights[label] for _, label in full_train_dataset]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Sample indices with class balance
        train_size = int(sample_ratio * len(full_train_dataset))
        train_indices = torch.multinomial(sample_weights, train_size, replacement=False)
        
        # Simple random sampling for validation
        val_size = int(sample_ratio * len(full_val_dataset))
        val_indices = torch.randperm(len(full_val_dataset))[:val_size]
        
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        
        logger.info(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        
        num_workers = 2 if torch.cuda.is_available() else 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader, len(full_train_dataset.classes)
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        return None, None, 0

def train_model(model_name, num_classes, batch_size=128, num_epochs=2, learning_rate=2e-3):
    """Fast training optimized for quick results with improved error handling"""
    logger.info(f"Starting training for {model_name}")
    start_time = time.time()
    
    try:
        is_skin_cancer = model_name == 'skin_cancer'
        data_dir = Path(f"backend/ml/data/{model_name}")
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
            
        train_loader, val_loader, actual_num_classes = create_data_loaders(
            data_dir, 
            batch_size=batch_size,
            is_skin_cancer=is_skin_cancer
        )
        
        if not train_loader or not val_loader:
            raise RuntimeError("Failed to create data loaders")
            
        # Verify we have the correct number of classes
        if actual_num_classes != num_classes:
            logger.warning(f"Expected {num_classes} classes but found {actual_num_classes} classes")
            num_classes = actual_num_classes
        
        # Use different models based on task with error handling
        try:
            if is_skin_cancer:
                model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
                learning_rate = 5e-4  # Lower learning rate for skin cancer
            else:
                model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=num_classes)
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise RuntimeError(f"Model creation failed: {str(e)}")
        
        model = model.to(device)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()
        
        # Optimizers with gradient clipping
        if is_skin_cancer:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=10,
                final_div_factor=100
            )
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2,
                div_factor=10,
                final_div_factor=100
            )
        
        best_acc = 0.0
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            
            for batch_idx, (inputs, targets) in enumerate(train_bar):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                for param in model.parameters():
                    param.grad = None
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                if is_skin_cancer:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                train_bar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.1f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.1e}'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad(), autocast():
                for inputs, targets in tqdm(val_loader, desc='Validation'):
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            logger.info(f"Train Acc: {100.*correct/total:.1f}%, Val Acc: {val_acc:.1f}%")
            logger.info(f"Train Loss: {running_loss/len(train_loader):.3f}, Val Loss: {val_loss/len(val_loader):.3f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save model with _model.pth suffix in the trained directory
                save_path = model_dir / f"{model_name}_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes,
                    'best_acc': best_acc,
                    'epoch': epoch
                }, save_path)
                logger.info(f"Saved best model to {save_path}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {best_acc:.1f}%")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def train_models():
    """Train all models with optimized settings for binary classification"""
    logger.info("Starting training pipeline")
    
    models_to_train = {
        'malaria': {
            'num_classes': 2,
            'batch_size': 128,
            'num_epochs': 3,
            'learning_rate': 2e-3
        },
        'skin_cancer': {
            'num_classes': 2,
            'batch_size': 32,
            'num_epochs': 3,
            'learning_rate': 5e-4
        }
    }
    
    for model_name, config in models_to_train.items():
        try:
            logger.info(f"\nTraining {model_name} model...")
            model = train_model(**config, model_name=model_name)
            
            if model is not None:
                # Save model with version control
                save_path = model_dir / f"{model_name}_final.pth"
                backup_path = model_dir / f"{model_name}_backup.pth"
                
                # Create backup of existing model if it exists
                if save_path.exists():
                    if backup_path.exists():
                        backup_path.unlink()
                    save_path.rename(backup_path)
                
                # Save new model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': config['num_classes'],
                    'training_completed': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, save_path)
                
                logger.info(f"Successfully saved {model_name} model")
            else:
                logger.error(f"Failed to train {model_name} model")
                
        except Exception as e:
            logger.error(f"Error training {model_name} model: {str(e)}")
            continue
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.info("Training pipeline completed!")

if __name__ == '__main__':
    train_models()
