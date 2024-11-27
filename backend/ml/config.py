from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "backend/ml/models/trained"
        
        # Create directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models directory: {self.models_dir}")
    
    def get_latest_model(self, model_type: str) -> Path:
        """Get the path to the latest model of the specified type.
        
        Args:
            model_type: Either 'skin_cancer' or 'malaria'
        """
        try:
            models = list(self.models_dir.glob(f"{model_type}*.pth"))
            if not models:
                logger.warning(f"No models found for type: {model_type}")
                return None
                
            # Sort by modification time to get the latest
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            logger.info(f"Latest {model_type} model: {latest_model}")
            return latest_model
            
        except Exception as e:
            logger.error(f"Error finding latest model for {model_type}: {str(e)}")
            return None
    
    def save_new_model(self, model_path: Path, model_type: str, fold: int = None) -> Path:
        """Save a newly trained model.
        
        Args:
            model_path: Path to the model file
            model_type: Either 'skin_cancer' or 'malaria'
            fold: Optional fold number for k-fold cross validation
        """
        try:
            if fold is not None:
                new_name = f"{model_type}_model_fold_{fold}.pth"
            else:
                new_name = f"{model_type}_model.pth"
            
            target_path = self.models_dir / new_name
            logger.info(f"Saving model to: {target_path}")
            
            # If file exists, remove it
            if target_path.exists():
                target_path.unlink()
                logger.info(f"Removed existing model: {target_path}")
            
            # Copy instead of move to avoid cross-device link errors
            model_path = Path(model_path)
            if model_path.exists():
                shutil.copy2(model_path, target_path)
                logger.info(f"Successfully saved model to: {target_path}")
            else:
                raise FileNotFoundError(f"Source model file not found: {model_path}")
                
            return target_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_type}: {str(e)}")
            raise
