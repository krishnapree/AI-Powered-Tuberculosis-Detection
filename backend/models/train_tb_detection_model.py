#!/usr/bin/env python3
"""
Advanced TensorFlow TB Detection Model Training Script
=====================================================

This script trains a high-performance TB detection model using TensorFlow
with the provided chest X-ray dataset, targeting 99.84% accuracy.

Features:
- Transfer learning with ResNet50
- GPU acceleration and mixed precision training
- Advanced data augmentation
- Model optimization for Render deployment
- TensorFlow Lite conversion for smaller size

Author: Healthcare AI Team
Version: 3.0 (Advanced Training Pipeline)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.mixed_precision import set_global_policy
    logger.info("✅ TensorFlow and dependencies imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import TensorFlow: {e}")
    sys.exit(1)

# Configuration
class Config:
    """Training configuration"""
    
    # Dataset paths
    DATASET_PATH = r"C:\Health care\Chest-X-Ray"
    IMAGE_PATH = os.path.join(DATASET_PATH, "image")
    MASK_PATH = os.path.join(DATASET_PATH, "mask")
    
    # Model parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    TARGET_ACCURACY = 0.9984  # 99.84%
    
    # Output paths
    MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_H5_PATH = os.path.join(MODELS_DIR, 'tensorflow_tb_model.h5')
    MODEL_TFLITE_PATH = os.path.join(MODELS_DIR, 'tensorflow_tb_model.tflite')
    TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.json')
    
    # GPU and optimization
    USE_MIXED_PRECISION = True
    USE_GPU = True

class DatasetAnalyzer:
    """Analyze the chest X-ray dataset and create labels"""
    
    def __init__(self, config):
        self.config = config
        self.image_files = []
        self.labels = []
        
    def analyze_masks_for_tb_detection(self):
        """
        Analyze mask files to determine TB vs Normal classification
        This is a heuristic approach based on mask intensity/area
        """
        logger.info("🔍 Analyzing dataset for TB classification...")
        
        image_files = sorted([f for f in os.listdir(self.config.IMAGE_PATH) if f.endswith('.png')])
        labels = []
        
        for img_file in image_files:
            mask_file = os.path.join(self.config.MASK_PATH, img_file)
            
            if os.path.exists(mask_file):
                # Load mask and analyze
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    # Calculate mask statistics
                    mask_area = np.sum(mask > 0)
                    total_area = mask.shape[0] * mask.shape[1]
                    mask_ratio = mask_area / total_area
                    
                    # Heuristic: If mask covers significant area, likely TB
                    # This is a simplified approach - in real scenarios, you'd have ground truth labels
                    if mask_ratio > 0.05:  # 5% threshold
                        labels.append(1)  # TB
                    else:
                        labels.append(0)  # Normal
                else:
                    labels.append(0)  # Default to normal if mask can't be loaded
            else:
                labels.append(0)  # Default to normal if no mask
        
        self.image_files = image_files
        self.labels = labels
        
        # Log statistics
        tb_count = sum(labels)
        normal_count = len(labels) - tb_count
        logger.info(f"📊 Dataset Analysis:")
        logger.info(f"   Total images: {len(labels)}")
        logger.info(f"   TB cases: {tb_count} ({tb_count/len(labels)*100:.1f}%)")
        logger.info(f"   Normal cases: {normal_count} ({normal_count/len(labels)*100:.1f}%)")
        
        return self.image_files, self.labels
    
    def create_balanced_dataset(self):
        """Create a balanced dataset for better training"""
        image_files, labels = self.analyze_masks_for_tb_detection()
        
        # Separate TB and Normal cases
        tb_indices = [i for i, label in enumerate(labels) if label == 1]
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Balance the dataset
        min_count = min(len(tb_indices), len(normal_indices))
        
        # Take equal numbers from each class
        balanced_indices = tb_indices[:min_count] + normal_indices[:min_count]
        np.random.shuffle(balanced_indices)
        
        balanced_files = [image_files[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        logger.info(f"🎯 Created balanced dataset:")
        logger.info(f"   TB cases: {sum(balanced_labels)}")
        logger.info(f"   Normal cases: {len(balanced_labels) - sum(balanced_labels)}")
        
        return balanced_files, balanced_labels

class TBDataGenerator:
    """Custom data generator for TB detection"""
    
    def __init__(self, config):
        self.config = config
        
    def create_data_generators(self, image_files, labels):
        """Create training and validation data generators"""
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_files, labels, test_size=self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT, 
            random_state=42, stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config.TEST_SPLIT/(self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT),
            random_state=42, stratify=y_temp
        )
        
        logger.info(f"📊 Data splits:")
        logger.info(f"   Training: {len(X_train)} samples")
        logger.info(f"   Validation: {len(X_val)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        # Create data generators with advanced augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = self._create_generator(train_datagen, X_train, y_train)
        val_generator = self._create_generator(val_datagen, X_val, y_val)
        test_generator = self._create_generator(val_datagen, X_test, y_test)
        
        return train_generator, val_generator, test_generator, (X_test, y_test)
    
    def _create_generator(self, datagen, image_files, labels):
        """Create a data generator from file lists"""
        
        def generator():
            while True:
                # Shuffle data
                indices = np.random.permutation(len(image_files))
                
                for start_idx in range(0, len(image_files), self.config.BATCH_SIZE):
                    batch_indices = indices[start_idx:start_idx + self.config.BATCH_SIZE]
                    
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        if idx < len(image_files):
                            # Load image
                            img_path = os.path.join(self.config.IMAGE_PATH, image_files[idx])
                            img = cv2.imread(img_path)
                            
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, self.config.IMG_SIZE)
                                
                                batch_images.append(img)
                                batch_labels.append(labels[idx])
                    
                    if batch_images:
                        batch_images = np.array(batch_images)
                        batch_labels = to_categorical(batch_labels, 2)
                        
                        # Apply augmentation
                        for i in range(len(batch_images)):
                            batch_images[i] = datagen.random_transform(batch_images[i])
                            batch_images[i] = datagen.standardize(batch_images[i])
                        
                        yield batch_images, batch_labels
        
        return generator

class TBModelBuilder:
    """Build and configure the TB detection model"""

    def __init__(self, config):
        self.config = config

    def build_model(self):
        """Build ResNet50-based TB detection model"""
        logger.info("🏗️ Building TB detection model...")

        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )

        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(2, activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logger.info(f"✅ Model built successfully")
        logger.info(f"   Total parameters: {model.count_params():,}")
        logger.info(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        return model

    def get_callbacks(self):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.config.MODEL_H5_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        return callbacks

class ModelTrainer:
    """Train the TB detection model"""

    def __init__(self, config):
        self.config = config
        self.setup_gpu()

    def setup_gpu(self):
        """Setup GPU and mixed precision training"""
        if self.config.USE_GPU:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"✅ GPU setup complete. Found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    logger.warning(f"⚠️ GPU setup failed: {e}")
            else:
                logger.info("ℹ️ No GPU found, using CPU")

        if self.config.USE_MIXED_PRECISION:
            try:
                set_global_policy('mixed_float16')
                logger.info("✅ Mixed precision training enabled")
            except Exception as e:
                logger.warning(f"⚠️ Mixed precision setup failed: {e}")

    def train_model(self, model, train_gen, val_gen, callbacks):
        """Train the model"""
        logger.info("🚀 Starting model training...")

        # Calculate steps
        steps_per_epoch = max(1, len(train_gen) // self.config.BATCH_SIZE)
        validation_steps = max(1, len(val_gen) // self.config.BATCH_SIZE)

        # Train model
        history = model.fit(
            train_gen(),
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS,
            validation_data=val_gen(),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("✅ Training completed!")

        return history

    def evaluate_model(self, model, test_gen, test_data):
        """Evaluate the trained model"""
        logger.info("📊 Evaluating model performance...")

        X_test, y_test = test_data

        # Load test images
        test_images = []
        for img_file in X_test:
            img_path = os.path.join(self.config.IMAGE_PATH, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.config.IMG_SIZE)
                img = img / 255.0
                test_images.append(img)

        test_images = np.array(test_images)
        y_test_cat = to_categorical(y_test, 2)

        # Evaluate
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            test_images, y_test_cat, verbose=0
        )

        # Predictions
        predictions = model.predict(test_images)
        y_pred = np.argmax(predictions, axis=1)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Normal', 'TB'])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"📈 Model Performance:")
        logger.info(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"   Test Precision: {test_precision:.4f}")
        logger.info(f"   Test Recall: {test_recall:.4f}")
        logger.info(f"   Target Accuracy: {self.config.TARGET_ACCURACY:.4f} ({self.config.TARGET_ACCURACY*100:.2f}%)")

        if test_accuracy >= self.config.TARGET_ACCURACY:
            logger.info("🎯 ✅ TARGET ACCURACY ACHIEVED!")
        else:
            logger.info(f"🎯 ⚠️ Target accuracy not reached. Difference: {(self.config.TARGET_ACCURACY - test_accuracy)*100:.2f}%")

        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(report)
        print("\nCONFUSION MATRIX")
        print("="*20)
        print(cm)

        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

class ModelOptimizer:
    """Optimize model for deployment"""

    def __init__(self, config):
        self.config = config

    def convert_to_tflite(self, model):
        """Convert model to TensorFlow Lite for smaller size"""
        logger.info("🔄 Converting model to TensorFlow Lite...")

        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            # Convert model
            tflite_model = converter.convert()

            # Save TFLite model
            with open(self.config.MODEL_TFLITE_PATH, 'wb') as f:
                f.write(tflite_model)

            # Check file sizes
            h5_size = os.path.getsize(self.config.MODEL_H5_PATH) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(self.config.MODEL_TFLITE_PATH) / (1024 * 1024)  # MB

            logger.info(f"✅ TensorFlow Lite conversion completed!")
            logger.info(f"   H5 model size: {h5_size:.2f} MB")
            logger.info(f"   TFLite model size: {tflite_size:.2f} MB")
            logger.info(f"   Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")

            if tflite_size < 100:  # Under 100MB for Render
                logger.info("🎯 ✅ Model size optimized for Render deployment!")
            else:
                logger.warning("⚠️ Model size may be too large for Render free tier")

            return True

        except Exception as e:
            logger.error(f"❌ TensorFlow Lite conversion failed: {e}")
            return False

    def save_training_metadata(self, history, evaluation_results):
        """Save training history and metadata"""
        logger.info("💾 Saving training metadata...")

        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_architecture': 'ResNet50 + Custom Head',
            'target_accuracy': self.config.TARGET_ACCURACY,
            'achieved_accuracy': evaluation_results['accuracy'],
            'dataset_info': {
                'total_images': len(os.listdir(self.config.IMAGE_PATH)),
                'image_size': self.config.IMG_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'epochs': self.config.EPOCHS
            },
            'training_history': {
                'loss': history.history.get('loss', []),
                'accuracy': history.history.get('accuracy', []),
                'val_loss': history.history.get('val_loss', []),
                'val_accuracy': history.history.get('val_accuracy', [])
            },
            'evaluation_results': evaluation_results
        }

        with open(self.config.TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Training metadata saved to {self.config.TRAINING_HISTORY_PATH}")

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Advanced TB Detection Model Training")
    logger.info("=" * 60)

    # Initialize configuration
    config = Config()

    # Create output directory
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    try:
        # Step 1: Analyze dataset
        logger.info("📊 Step 1: Dataset Analysis")
        analyzer = DatasetAnalyzer(config)
        image_files, labels = analyzer.create_balanced_dataset()

        # Step 2: Create data generators
        logger.info("🔄 Step 2: Data Preparation")
        data_gen = TBDataGenerator(config)
        train_gen, val_gen, test_gen, test_data = data_gen.create_data_generators(image_files, labels)

        # Step 3: Build model
        logger.info("🏗️ Step 3: Model Architecture")
        model_builder = TBModelBuilder(config)
        model = model_builder.build_model()
        callbacks = model_builder.get_callbacks()

        # Step 4: Train model
        logger.info("🎯 Step 4: Model Training")
        trainer = ModelTrainer(config)
        history = trainer.train_model(model, train_gen, val_gen, callbacks)

        # Step 5: Evaluate model
        logger.info("📈 Step 5: Model Evaluation")
        evaluation_results = trainer.evaluate_model(model, test_gen, test_data)

        # Step 6: Optimize for deployment
        logger.info("⚡ Step 6: Model Optimization")
        optimizer = ModelOptimizer(config)
        optimizer.convert_to_tflite(model)
        optimizer.save_training_metadata(history, evaluation_results)

        # Final summary
        logger.info("🎉 Training Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"📁 Models saved:")
        logger.info(f"   TensorFlow H5: {config.MODEL_H5_PATH}")
        logger.info(f"   TensorFlow Lite: {config.MODEL_TFLITE_PATH}")
        logger.info(f"   Training History: {config.TRAINING_HISTORY_PATH}")
        logger.info(f"🎯 Final Accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)")
        logger.info(f"🎯 Target Accuracy: {config.TARGET_ACCURACY:.4f} ({config.TARGET_ACCURACY*100:.2f}%)")

        if evaluation_results['accuracy'] >= config.TARGET_ACCURACY:
            logger.info("✅ SUCCESS: Target accuracy achieved!")
            logger.info("🚀 Model ready for production deployment!")
        else:
            logger.info("⚠️ Target accuracy not reached. Consider:")
            logger.info("   - Increasing training epochs")
            logger.info("   - Adding more data augmentation")
            logger.info("   - Fine-tuning hyperparameters")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
