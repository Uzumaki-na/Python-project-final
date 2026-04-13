import os
import kaggle
import zipfile
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)

# Get the absolute path to the data directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

def download_skin_cancer_dataset():
    """
    Downloads the ISIC 2020 skin cancer dataset from Kaggle
    Dataset: https://www.kaggle.com/datasets/nroman/melanoma-external-malignant-256
    """
    logging.info("Downloading skin cancer dataset...")
    
    # Create data directory if it doesn't exist
    skin_cancer_dir = DATA_DIR / 'skin_cancer'
    skin_cancer_dir.mkdir(exist_ok=True)
    
    try:
        # Download dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'nroman/melanoma-external-malignant-256',
            path=str(skin_cancer_dir),
            unzip=True
        )
        logging.info("Successfully downloaded skin cancer dataset")
        
        # Organize into train/val directories
        train_dir = skin_cancer_dir / 'train'
        val_dir = skin_cancer_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        for class_name in ['benign', 'malignant']:
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            
    except Exception as e:
        logging.error(f"Error downloading skin cancer dataset: {str(e)}")
        raise

def download_malaria_dataset():
    """
    Downloads the malaria cell images dataset from Kaggle
    Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
    """
    logging.info("Downloading malaria dataset...")
    
    # Create data directory if it doesn't exist
    malaria_dir = DATA_DIR / 'malaria'
    malaria_dir.mkdir(exist_ok=True)
    
    try:
        # Download dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'iarunava/cell-images-for-detecting-malaria',
            path=str(malaria_dir),
            unzip=True
        )
        logging.info("Successfully downloaded malaria dataset")
        
        # Organize into train/val directories
        train_dir = malaria_dir / 'train'
        val_dir = malaria_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        for class_name in ['Parasitized', 'Uninfected']:
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            
    except Exception as e:
        logging.error(f"Error downloading malaria dataset: {str(e)}")
        raise

def prepare_datasets():
    """
    Prepares both datasets for training
    """
    try:
        # Download datasets
        download_skin_cancer_dataset()
        download_malaria_dataset()
        
        logging.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logging.error(f"Failed to prepare datasets: {str(e)}")
        raise

if __name__ == '__main__':
    prepare_datasets()
