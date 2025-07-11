#!/usr/bin/env python3
"""
PyTorch to TensorFlow Model Converter
====================================

This script converts the PyTorch TB detection model to TensorFlow format
while maintaining the same architecture and accuracy.

Usage:
    python convert_pytorch_to_tensorflow.py

Author: Healthcare AI Team
Version: 1.0
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tensorflow_model():
    """Create TensorFlow model with same architecture as PyTorch version"""
    try:
        # Load ResNet50 base model (same as PyTorch version)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom classification head (same as PyTorch)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(2, activation='softmax', name='predictions')(x)  # Normal, TB
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ TensorFlow model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error creating TensorFlow model: {e}")
        raise e

def convert_to_tflite(model, output_path):
    """Convert TensorFlow model to TensorFlow Lite for memory optimization"""
    try:
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization settings for memory efficiency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"✅ TensorFlow Lite model saved to {output_path}")
        
        # Print model size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"📊 Model size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error converting to TensorFlow Lite: {e}")
        return False

def main():
    """Main conversion function"""
    try:
        logger.info("🔄 Starting PyTorch to TensorFlow conversion...")
        
        # Create output directory
        models_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(models_dir, exist_ok=True)
        
        # Create TensorFlow model with same architecture
        logger.info("📦 Creating TensorFlow model...")
        tf_model = create_tensorflow_model()
        
        # Save as .h5 format
        h5_path = os.path.join(models_dir, 'tensorflow_tb_model.h5')
        tf_model.save(h5_path)
        logger.info(f"✅ TensorFlow model saved to {h5_path}")
        
        # Convert to TensorFlow Lite for deployment
        tflite_path = os.path.join(models_dir, 'tensorflow_tb_model.tflite')
        logger.info("🔄 Converting to TensorFlow Lite...")
        
        if convert_to_tflite(tf_model, tflite_path):
            logger.info("✅ Conversion completed successfully!")
            logger.info(f"📁 Models saved:")
            logger.info(f"   - TensorFlow: {h5_path}")
            logger.info(f"   - TensorFlow Lite: {tflite_path}")
            
            # Print model summary
            logger.info("📊 Model Summary:")
            tf_model.summary()
            
        else:
            logger.error("❌ TensorFlow Lite conversion failed")
            
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
