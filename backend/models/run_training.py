#!/usr/bin/env python3
"""
TB Detection Model Training Runner
=================================

This script runs the TB detection model training with proper environment setup
and error handling.

Usage:
    python run_training.py

Author: Healthcare AI Team
Version: 1.0
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'tensorflow',
        'opencv-python',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_dataset():
    """Check if the dataset exists and is accessible"""
    dataset_path = r"C:\Health care\Chest-X-Ray"
    image_path = os.path.join(dataset_path, "image")
    mask_path = os.path.join(dataset_path, "mask")
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image directory not found: {image_path}")
        return False
    
    if not os.path.exists(mask_path):
        logger.error(f"❌ Mask directory not found: {mask_path}")
        return False
    
    # Count files
    image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
    
    logger.info(f"✅ Dataset found:")
    logger.info(f"   Images: {len(image_files)}")
    logger.info(f"   Masks: {len(mask_files)}")
    
    if len(image_files) == 0:
        logger.error("❌ No image files found")
        return False
    
    return True

def run_training():
    """Run the training script"""
    logger.info("🚀 Starting TB Detection Model Training...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, 'train_tb_detection_model.py')
    
    if not os.path.exists(training_script):
        logger.error(f"❌ Training script not found: {training_script}")
        return False
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, training_script], 
                              capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            logger.info("✅ Training completed successfully!")
            logger.info("Training output:")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Training failed!")
            logger.error("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run training: {e}")
        return False

def main():
    """Main execution function"""
    logger.info("🔧 TB Detection Model Training Setup")
    logger.info("=" * 50)
    
    # Step 1: Check dependencies
    logger.info("📦 Step 1: Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    # Step 2: Check dataset
    logger.info("📊 Step 2: Checking dataset...")
    if not check_dataset():
        logger.error("❌ Dataset check failed")
        sys.exit(1)
    
    # Step 3: Run training
    logger.info("🎯 Step 3: Running training...")
    if not run_training():
        logger.error("❌ Training failed")
        sys.exit(1)
    
    logger.info("🎉 All steps completed successfully!")
    logger.info("🚀 Your TB detection model is ready for deployment!")

if __name__ == "__main__":
    main()
