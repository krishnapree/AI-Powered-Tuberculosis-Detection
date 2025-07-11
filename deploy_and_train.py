#!/usr/bin/env python3
"""
Deployment and Training Script for Render
=========================================

This script will run on Render to automatically train the TB detection model
with the provided dataset and deploy the trained model.

Author: Healthcare AI Team
Version: 1.0
"""

import os
import sys
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for training on Render"""
    logger.info("🔧 Setting up environment for TB model training...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 11:
        logger.info("✅ Python version compatible with TensorFlow")
        return True
    else:
        logger.warning(f"⚠️ Python {python_version.major}.{python_version.minor} may have TensorFlow compatibility issues")
        return True  # Continue anyway

def install_training_dependencies():
    """Install dependencies required for training"""
    logger.info("📦 Installing training dependencies...")
    
    training_packages = [
        'tensorflow==2.13.0',
        'opencv-python-headless==4.8.1.78',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2',
        'seaborn==0.12.2'
    ]
    
    for package in training_packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_sample_dataset():
    """Create a sample dataset structure for training demonstration"""
    logger.info("📊 Creating sample dataset for training...")
    
    # This would be replaced with your actual dataset in production
    dataset_path = "sample_dataset"
    os.makedirs(os.path.join(dataset_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "mask"), exist_ok=True)
    
    # In production, you would upload your actual dataset here
    logger.info("💡 Note: In production, upload your chest X-ray dataset to this location")
    
    return dataset_path

def run_model_training():
    """Run the TB detection model training"""
    logger.info("🚀 Starting TB detection model training...")
    
    try:
        # Navigate to models directory
        models_dir = os.path.join(os.getcwd(), 'backend', 'models')
        
        if os.path.exists(os.path.join(models_dir, 'train_tb_detection_model.py')):
            # Run training script
            result = subprocess.run([
                sys.executable, 
                os.path.join(models_dir, 'train_tb_detection_model.py')
            ], capture_output=True, text=True, cwd=models_dir)
            
            if result.returncode == 0:
                logger.info("✅ Model training completed successfully!")
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
                return False
        else:
            logger.warning("⚠️ Training script not found, using conversion script...")
            # Fallback to conversion script
            result = subprocess.run([
                sys.executable, 
                os.path.join(models_dir, 'convert_pytorch_to_tensorflow.py')
            ], capture_output=True, text=True, cwd=models_dir)
            
            if result.returncode == 0:
                logger.info("✅ Model conversion completed successfully!")
                return True
            else:
                logger.error(f"❌ Model conversion failed: {result.stderr}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Training process failed: {e}")
        return False

def verify_model_files():
    """Verify that model files were created successfully"""
    logger.info("🔍 Verifying model files...")
    
    models_dir = os.path.join(os.getcwd(), 'backend', 'models')
    
    required_files = [
        'tensorflow_tb_model.h5',
        'tensorflow_tb_model.tflite'
    ]
    
    all_files_exist = True
    for file_name in required_files:
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"✅ {file_name}: {file_size:.2f} MB")
        else:
            logger.error(f"❌ Missing: {file_name}")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Main deployment and training function"""
    logger.info("🚀 TB Detection Model Deployment and Training")
    logger.info("=" * 60)
    
    try:
        # Step 1: Setup environment
        if not setup_environment():
            logger.error("❌ Environment setup failed")
            sys.exit(1)
        
        # Step 2: Install dependencies
        if not install_training_dependencies():
            logger.error("❌ Dependency installation failed")
            # Continue anyway for basic deployment
            logger.info("⚠️ Continuing with basic deployment...")
        
        # Step 3: Run training (or conversion)
        if not run_model_training():
            logger.error("❌ Model training/conversion failed")
            sys.exit(1)
        
        # Step 4: Verify model files
        if not verify_model_files():
            logger.error("❌ Model verification failed")
            sys.exit(1)
        
        logger.info("🎉 Deployment and training completed successfully!")
        logger.info("🚀 TB Detection model is ready for production!")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
