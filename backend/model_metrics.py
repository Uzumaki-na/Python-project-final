"""
Model Metrics and Accuracy Evaluation Script
Evaluates both Skin Cancer and Malaria models on validation data
"""
import torch
import timm
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Model paths
MODELS_DIR = Path("c:/Users/panav/Downloads/Python-project-final-main/Python-project-final-main/backend/ml/models/trained")
DATA_DIR = Path("c:/Users/panav/Downloads/Python-project-final-main/Python-project-final-main/backend/ml/data")

def load_model(model_name, model_type):
    """Load a trained model"""
    if model_type == 'skin_cancer':
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        model_path = MODELS_DIR / "skin_cancer_final.pth"
        target_size = (128, 128)
    else:  # malaria
        model = timm.create_model('efficientnet_lite0', pretrained=False, num_classes=2)
        model_path = MODELS_DIR / "malaria_model.pth"
        target_size = (96, 96)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Get training info
    best_acc = checkpoint.get('best_acc', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    
    return model, target_size, best_acc, epoch

def get_validation_data(model_type, target_size):
    """Get validation data paths and labels"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_paths = []
    labels = []
    
    if model_type == 'skin_cancer':
        # Skin cancer validation data
        val_dir = DATA_DIR / "skin_cancer" / "val"
        if val_dir.exists():
            for class_idx, class_name in enumerate(['benign', 'malignant']):
                class_dir = val_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.jpg"):
                        data_paths.append((str(img_path), class_idx, class_name))
    else:  # malaria
        # Malaria validation data
        val_dir = DATA_DIR / "malaria" / "val"
        if val_dir.exists():
            for class_idx, class_name in enumerate(['Parasitized', 'Uninfected']):
                class_dir = val_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.png"):
                        data_paths.append((str(img_path), class_idx, class_name))
    
    return data_paths, transform

def evaluate_model(model, data_paths, transform, model_name, class_names):
    """Evaluate model and return metrics"""
    y_true = []
    y_pred = []
    y_probs = []
    
    print(f"\nEvaluating {model_name} on {len(data_paths)} validation samples...")
    
    with torch.no_grad():
        for img_path, true_label, class_name in tqdm(data_paths, desc=f"Evaluating {model_name}"):
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_label = torch.argmax(probabilities).item()
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_probs.append(probabilities.cpu().numpy())
    
    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names
    }

def print_metrics(model_name, metrics, checkpoint_acc, epoch):
    """Print formatted metrics"""
    print(f"\n{'='*60}")
    print(f"  {model_name} MODEL METRICS")
    print(f"{'='*60}")
    print(f"\n  Training Checkpoint Info:")
    print(f"    - Best Validation Accuracy (from training): {checkpoint_acc}")
    print(f"    - Epoch: {epoch}")
    print(f"\n  Validation Metrics:")
    print(f"    - Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    - Precision: {metrics['precision']*100:.2f}%")
    print(f"    - Recall:    {metrics['recall']*100:.2f}%")
    print(f"    - F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                      Predicted")
    print(f"                 {metrics['class_names'][0]:<12} {metrics['class_names'][1]:<12}")
    print(f"  Actual {metrics['class_names'][0]:<8} {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"         {metrics['class_names'][1]:<8} {cm[1][0]:<12} {cm[1][1]:<12}")
    
    # Classification report
    print(f"\n  Detailed Classification Report:")
    print(classification_report(metrics['y_true'], metrics['y_pred'], 
                               target_names=metrics['class_names'], digits=4))
    print(f"{'='*60}\n")

def main():
    print("\n" + "="*60)
    print("  MEDICAL AI DIAGNOSTIC - MODEL EVALUATION")
    print("="*60)
    
    # Evaluate Skin Cancer Model
    print("\n\n[1] LOADING SKIN CANCER MODEL...")
    try:
        skin_model, skin_size, skin_ckpt_acc, skin_epoch = load_model('skin_cancer', 'skin_cancer')
        print(f"    ✓ Model loaded successfully")
        print(f"    - Architecture: EfficientNet-B0")
        print(f"    - Input size: {skin_size}")
        
        skin_data, skin_transform = get_validation_data('skin_cancer', skin_size)
        if skin_data:
            skin_metrics = evaluate_model(skin_model, skin_data, skin_transform, 
                                         "Skin Cancer", ['Benign', 'Malignant'])
            print_metrics("SKIN CANCER", skin_metrics, f"{skin_ckpt_acc:.2f}%" if isinstance(skin_ckpt_acc, float) else skin_ckpt_acc, 
                         skin_epoch if skin_epoch != 'N/A' else "N/A")
        else:
            print("    ✗ No validation data found for Skin Cancer")
    except Exception as e:
        print(f"    ✗ Error evaluating Skin Cancer model: {e}")
    
    # Evaluate Malaria Model
    print("\n\n[2] LOADING MALARIA MODEL...")
    try:
        malaria_model, malaria_size, malaria_ckpt_acc, malaria_epoch = load_model('malaria', 'malaria')
        print(f"    ✓ Model loaded successfully")
        print(f"    - Architecture: EfficientNet-Lite0")
        print(f"    - Input size: {malaria_size}")
        
        malaria_data, malaria_transform = get_validation_data('malaria', malaria_size)
        if malaria_data:
            malaria_metrics = evaluate_model(malaria_model, malaria_data, malaria_transform,
                                           "Malaria", ['Parasitized', 'Uninfected'])
            print_metrics("MALARIA", malaria_metrics, f"{malaria_ckpt_acc:.2f}%" if isinstance(malaria_ckpt_acc, float) else malaria_ckpt_acc,
                         malaria_epoch if malaria_epoch != 'N/A' else "N/A")
        else:
            print("    ✗ No validation data found for Malaria")
    except Exception as e:
        print(f"    ✗ Error evaluating Malaria model: {e}")
    
    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
