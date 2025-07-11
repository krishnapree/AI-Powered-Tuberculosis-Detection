# TB Detection Models

This directory contains the machine learning models for tuberculosis detection.

## Model Conversion

The project has been converted from PyTorch to TensorFlow for better memory efficiency and deployment compatibility.

### Available Models

1. **tensorflow_tb_model.h5** - Full TensorFlow model
2. **tensorflow_tb_model.tflite** - Optimized TensorFlow Lite model (recommended for deployment)

### Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Head**: 
  - GlobalAveragePooling2D
  - Dropout(0.5)
  - Dense(512, activation='relu')
  - Dropout(0.3)
  - Dense(2, activation='softmax') # Normal, Tuberculosis
- **Input Size**: 224x224x3
- **Accuracy**: 99.84%

### Converting Models

To convert PyTorch models to TensorFlow:

```bash
cd backend/models
python convert_pytorch_to_tensorflow.py
```

### Model Usage

The TensorFlow Lite model is automatically loaded by the TB detection service for memory-optimized inference.

### Memory Optimization

- TensorFlow Lite reduces model size by ~75%
- Float16 precision for smaller memory footprint
- Lazy loading - model loaded only when needed
- Automatic memory cleanup after inference
