import os
import shutil
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectCleaner:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "backend" / "ml" / "models"
        self.trained_dir = self.models_dir / "trained"
        
    def cleanup_cache(self):
        """Remove all __pycache__ directories"""
        logger.info("Cleaning up Python cache files...")
        for pycache in self.project_root.rglob("__pycache__"):
            if pycache.is_dir():
                shutil.rmtree(pycache)
                logger.info(f"Removed {pycache}")
                
    def cleanup_logs(self):
        """Remove log files except the most recent ones"""
        logger.info("Cleaning up log files...")
        log_files = list(self.project_root.rglob("*.log"))
        current_time = time.time()
        
        for log_file in log_files:
            # Keep logs that are less than 7 days old
            if current_time - log_file.stat().st_mtime > 7 * 24 * 3600:
                log_file.unlink()
                logger.info(f"Removed old log: {log_file}")
                
    def cleanup_temp(self):
        """Remove temporary directories and files"""
        temp_dirs = [
            self.project_root / "temp_downloads",
            self.project_root / "runs",
            self.project_root / ".pytest_cache",
            self.project_root / "frontend" / "node_modules",
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temporary directory: {temp_dir}")
                
    def organize_models(self):
        """Organize model files into a clean structure"""
        logger.info("Organizing model files...")
        
        # Create directories if they don't exist
        model_dirs = {
            "production": self.trained_dir / "production",
            "backup": self.trained_dir / "backup",
            "archive": self.trained_dir / "archive"
        }
        
        for dir_path in model_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Move model files to appropriate directories
        model_patterns = {
            "malaria": ["malaria_*.pth"],
            "skin_cancer": ["skin_cancer_*.pth", "best_skin_cancer_*.pth"]
        }
        
        for model_type, patterns in model_patterns.items():
            for pattern in patterns:
                model_files = list(self.models_dir.glob(pattern))
                if not model_files:
                    continue
                    
                # Sort by modification time (newest first)
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Move the best/latest model to production
                best_model = model_files[0]
                new_name = f"{model_type}_model.pth"
                shutil.copy2(best_model, model_dirs["production"] / new_name)
                logger.info(f"Copied best {model_type} model to production: {new_name}")
                
                # Move the second best to backup (if exists)
                if len(model_files) > 1:
                    backup_model = model_files[1]
                    new_name = f"{model_type}_backup.pth"
                    shutil.copy2(backup_model, model_dirs["backup"] / new_name)
                    logger.info(f"Copied backup {model_type} model: {new_name}")
                
                # Move remaining files to archive
                for old_model in model_files[2:]:
                    archive_name = f"{model_type}_archive_{old_model.stat().st_mtime}.pth"
                    shutil.move(old_model, model_dirs["archive"] / archive_name)
                    logger.info(f"Archived old model: {archive_name}")
                    
    def cleanup_all(self):
        """Run all cleanup operations"""
        try:
            logger.info("Starting project cleanup...")
            self.cleanup_cache()
            self.cleanup_logs()
            self.cleanup_temp()
            self.organize_models()
            logger.info("Project cleanup completed successfully!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    cleaner = ProjectCleaner(project_root)
    cleaner.cleanup_all()
