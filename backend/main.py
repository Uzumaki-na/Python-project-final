import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
from ml.models.models import SkinCancerModel, MalariaModel
from ml.data.datasets import preprocess_image
from ml.config import ModelConfig
from pathlib import Path
import time
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Diagnostic API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model config
model_config = ModelConfig()

# Global variables for models and their status
models: Dict[str, Optional[torch.nn.Module]] = {
    "skin_cancer": None,
    "malaria": None
}
model_load_times: Dict[str, float] = {}

def load_model(model_class, model_path: Path, model_name: str) -> bool:
    """Load a model with proper error handling and logging"""
    try:
        if model_path is None:
            logger.error(f"No model file found for {model_name}")
            return False
            
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        model = model_class()
        model = model.to('cpu')  # Ensure model is on CPU
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False

        # Try different state dict formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try loading with different strategies
        try:
            # First try direct loading
            model.load_state_dict(state_dict)
        except Exception as e1:
            try:
                # Try loading without strict matching
                model.load_state_dict(state_dict, strict=False)
                logger.warning(f"Loaded {model_name} model with strict=False")
            except Exception as e2:
                # Try loading with model prefix
                try:
                    prefixed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
                    model.load_state_dict(prefixed_state_dict, strict=False)
                    logger.warning(f"Loaded {model_name} model with 'model.' prefix")
                except Exception as e3:
                    logger.error(f"Failed to load {model_name} model: {str(e3)}")
                    return False

        model.eval()  # Set to evaluation mode
        models[model_name] = model
        model_load_times[model_name] = time.time()
        logger.info(f"Successfully loaded {model_name} model")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {model_name} model from {model_path}: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up the server...")
    
    # Define specific model paths
    skin_cancer_path = model_config.models_dir / "skin_cancer_final.pth"
    malaria_path = model_config.models_dir / "malaria_model.pth"
    
    logger.info(f"Loading skin cancer model from: {skin_cancer_path}")
    logger.info(f"Loading malaria model from: {malaria_path}")
    
    # Load models
    if not load_model(SkinCancerModel, skin_cancer_path, "skin_cancer"):
        logger.error("Failed to load skin cancer model")
    
    if not load_model(MalariaModel, malaria_path, "malaria"):
        logger.error("Failed to load malaria model")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical AI Diagnostic API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict/skin-cancer",
            "/predict/malaria"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            name: {
                "loaded": model is not None,
                "last_loaded": time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(model_load_times.get(name, 0)))
            }
            for name, model in models.items()
        },
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    }

def process_image_file(file: UploadFile, model_type: str = 'malaria'):
    """Process uploaded image file for model inference"""
    try:
        # Read image file
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image for the specific model type
        image_tensor = preprocess_image(image, model_type=model_type)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    finally:
        file.file.close()

@app.post("/predict/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):
    """Endpoint for skin cancer prediction"""
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    
    try:
        # Process image
        image_tensor = process_image_file(file, model_type='skin_cancer')
        
        # Get prediction
        models["skin_cancer"].eval()  # Ensure model is in eval mode
        with torch.no_grad():
            outputs = models["skin_cancer"](image_tensor)
            # Log raw outputs for debugging
            logger.info(f"Raw model outputs: {outputs}")
            
            # Apply softmax properly
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            logger.info(f"After softmax: {probabilities}")
            
            # Convert to standard Python floats
            benign_prob = float(probabilities[0].item())
            malignant_prob = float(probabilities[1].item())
            
            # Ensure probabilities sum to 1
            total = benign_prob + malignant_prob
            if total > 0:
                benign_prob = benign_prob / total
                malignant_prob = malignant_prob / total
            
            # Double check the values
            logger.info(f"Final probs - Benign: {benign_prob}, Malignant: {malignant_prob}")
            
            # Determine prediction based on higher probability
            prediction = 0 if benign_prob > malignant_prob else 1
            confidence = benign_prob if prediction == 0 else malignant_prob
            
        predicted_class = "Benign" if prediction == 0 else "Malignant"
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 3),  # Raw probability
            "probabilities": {
                "Benign": round(benign_prob, 3),
                "Malignant": round(malignant_prob, 3)
            },
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error predicting skin cancer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/malaria")
async def predict_malaria(file: UploadFile = File(...)):
    """Endpoint for malaria prediction"""
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    
    try:
        # Process image
        image_tensor = process_image_file(file, model_type='malaria')
        
        # Get prediction
        models["malaria"].eval()  # Ensure model is in eval mode
        with torch.no_grad():
            outputs = models["malaria"](image_tensor)
            # Log raw outputs for debugging
            logger.info(f"Raw model outputs: {outputs}")
            
            # Apply softmax properly
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            logger.info(f"After softmax: {probabilities}")
            
            # Convert to standard Python floats
            parasitized_prob = float(probabilities[0].item())
            uninfected_prob = float(probabilities[1].item())
            
            # Ensure probabilities sum to 1
            total = parasitized_prob + uninfected_prob
            if total > 0:
                parasitized_prob = parasitized_prob / total
                uninfected_prob = uninfected_prob / total
            
            # Double check the values
            logger.info(f"Final probs - Parasitized: {parasitized_prob}, Uninfected: {uninfected_prob}")
            
            # Determine prediction based on higher probability
            prediction = 0 if parasitized_prob > uninfected_prob else 1
            confidence = parasitized_prob if prediction == 0 else uninfected_prob
            
        predicted_class = "Parasitized" if prediction == 0 else "Uninfected"
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 3),  # Raw probability
            "probabilities": {
                "Parasitized": round(parasitized_prob, 3),
                "Uninfected": round(uninfected_prob, 3)
            },
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error predicting malaria: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
