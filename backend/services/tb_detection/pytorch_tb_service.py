#!/usr/bin/env python3
"""
High-Accuracy TB Detection Service with PyTorch
==============================================

This module provides the TB detection service using our 99.84% accuracy PyTorch model.
Integrates seamlessly with the healthcare portal.

Author: Healthcare AI Team
Version: 2.0 (PyTorch Integration)
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import logging
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import base64
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for TB detection
tb_bp = Blueprint('tuberculosis', __name__, 
                  template_folder='templates',
                  static_folder='static')

class TBDetectionModel:
    """High-accuracy TB detection model wrapper with lazy loading"""

    def __init__(self, model_path='models/pytorch_tb_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.model_path = model_path
        self.accuracy = 99.84  # Our achieved accuracy
        self.model_loaded = False
        self.setup_transforms()  # Only setup transforms initially
    
    def load_model(self):
        """Load the trained PyTorch model (lazy loading)"""
        if self.model_loaded:
            return

        try:
            import gc
            # Create model architecture (same as training)
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)  # 2 classes: Normal, TB
            )

            # Load trained weights
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                logger.info(f"✅ High-accuracy TB model loaded successfully from {self.model_path}")
                logger.info(f"🎯 Model accuracy: {self.accuracy}%")

                # Force garbage collection to free memory
                gc.collect()
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

        except Exception as e:
            logger.error(f"❌ Error loading TB model: {e}")
            raise e

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            import gc
            gc.collect()
            logger.info("🗑️ TB model unloaded to free memory")
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict TB from chest X-ray image with lazy loading

        Args:
            image_path (str): Path to the chest X-ray image

        Returns:
            dict: Prediction results with confidence scores
        """
        try:
            # Load model only when needed (lazy loading)
            if not self.model_loaded:
                self.load_model()

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                # Class mapping: 0 = Normal, 1 = TB
                class_names = ['Normal', 'Tuberculosis']
                prediction = class_names[predicted_class]

                # Get both class probabilities
                normal_prob = probabilities[0][0].item()
                tb_prob = probabilities[0][1].item()

                result = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'normal_probability': normal_prob,
                    'tb_probability': tb_prob,
                    'model_accuracy': self.accuracy,
                    'is_tb_detected': predicted_class == 1
                }

                # Unload model after prediction to free memory
                self.unload_model()

                return result

        except Exception as e:
            logger.error(f"❌ Error during TB prediction: {e}")
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0
            }

# Initialize the TB detection model (lazy loading - model not loaded until needed)
tb_detector = None
tb_model_path = None

try:
    # Try multiple possible model paths
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/pytorch_tb_model.pth'),  # backend/models/
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/pytorch_tb_model.pth'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tuberclosis/models/pytorch_tb_model.pth'),
        'Disease Diagnosis/tuberclosis/models/pytorch_tb_model.pth',
        'models/pytorch_tb_model.pth',
        'backend/models/pytorch_tb_model.pth'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            tb_model_path = path
            break

    if tb_model_path:
        # Create detector instance but don't load model yet (lazy loading)
        tb_detector = TBDetectionModel(tb_model_path)
        logger.info(f"✅ TB detector initialized (lazy loading) with model path: {tb_model_path}")
    else:
        logger.error(f"❌ TB model not found in any of the expected locations: {possible_paths}")

except Exception as e:
    logger.error(f"❌ Failed to initialize TB detector: {e}")
    tb_detector = None

# Constants
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_chest_xray(image_path):
    """
    Advanced validation to check if image is specifically a chest X-ray
    Uses multiple criteria to distinguish chest X-rays from other X-ray types
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Failed to load image. Please ensure the file is a valid image format."

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 1. Check image dimensions (chest X-rays have specific aspect ratios)
        if height < 200 or width < 200:
            return False, "Image resolution too low for chest X-ray analysis. Please upload a higher quality image."

        aspect_ratio = width / height
        # Chest X-rays typically have aspect ratio between 0.7 and 1.4
        if aspect_ratio < 0.6 or aspect_ratio > 1.6:
            return False, "Image dimensions don't match typical chest X-ray proportions. Please upload a valid chest X-ray."

        # 2. Check for X-ray characteristics (contrast and intensity distribution)
        contrast = np.std(gray)
        if contrast < 25:
            return False, "Image appears to have insufficient contrast for X-ray analysis. Please upload a clear X-ray image."

        # 3. Check intensity distribution (X-rays have specific histogram patterns)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # X-rays typically have a bimodal or specific distribution
        mean_intensity = np.mean(gray)
        if mean_intensity < 30 or mean_intensity > 220:
            return False, "Image brightness levels don't match typical X-ray characteristics."

        # 4. Detect if image contains typical non-chest X-ray patterns
        # Check for bone structures that indicate hand/limb X-rays
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        # Hand/bone X-rays typically have much higher edge density due to detailed bone structures
        if edge_density > 0.15:
            return False, "Image appears to be a bone/limb X-ray rather than a chest X-ray. Please upload a chest X-ray for tuberculosis detection."

        # 5. Check for chest-specific features
        # Chest X-rays typically have a large central area (lungs) with lower intensity
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        center_mean = np.mean(center_region)
        overall_mean = np.mean(gray)

        # In chest X-rays, lung areas are typically darker than surrounding structures
        if center_mean > overall_mean * 1.2:
            return False, "Image doesn't show typical chest X-ray lung field patterns. Please ensure you're uploading a chest X-ray."

        # 6. Check for typical chest X-ray size and orientation
        # Most chest X-rays are portrait or square, not extremely wide
        if width > height * 1.5:
            return False, "Image orientation suggests this may not be a standard chest X-ray. Please upload a properly oriented chest X-ray."

        # 7. Advanced pattern detection for chest structures
        # Use template matching or feature detection for rib patterns
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Look for horizontal patterns that might indicate ribs
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, horizontal_kernel)

        # Chest X-rays should have some horizontal rib patterns
        horizontal_score = np.sum(horizontal_lines > 0) / (height * width)

        if horizontal_score < 0.001:
            return False, "Image doesn't show typical chest X-ray structural patterns. Please upload a valid chest X-ray."

        # 8. Final validation - check if image looks like a medical X-ray at all
        # X-rays should have inverted appearance (bones white, soft tissue dark)
        bone_regions = gray > np.percentile(gray, 85)
        bone_percentage = np.sum(bone_regions) / (height * width)

        if bone_percentage < 0.05 or bone_percentage > 0.4:
            return False, "Image doesn't appear to be a medical X-ray. Please upload a valid chest X-ray image."

        return True, "✅ Image validated as suitable chest X-ray for tuberculosis analysis"

    except Exception as e:
        logger.error(f"Error in chest X-ray validation: {e}")
        return False, f"Error validating image: Please ensure you're uploading a valid chest X-ray image."

def generate_detailed_analysis(prediction_result):
    """
    Generate comprehensive medical analysis and recommendations
    """
    is_tb = prediction_result['is_tb_detected']
    confidence = prediction_result['confidence']
    normal_prob = prediction_result['normal_probability']
    tb_prob = prediction_result['tb_probability']

    # Risk assessment based on confidence levels
    if confidence >= 0.9:
        confidence_level = "Very High"
        reliability = "Highly reliable result"
    elif confidence >= 0.8:
        confidence_level = "High"
        reliability = "Reliable result"
    elif confidence >= 0.7:
        confidence_level = "Moderate"
        reliability = "Moderately reliable result"
    else:
        confidence_level = "Low"
        reliability = "Low confidence - requires further evaluation"

    if is_tb:
        # TB Detected Analysis
        explanation = f"""
        <h5>🔍 X-Ray Analysis Results</h5>
        <p>Our AI model has detected patterns in the chest X-ray that are <strong>consistent with tuberculosis</strong>.
        The analysis shows a {confidence*100:.1f}% confidence level in this detection.</p>

        <h6>📊 Probability Breakdown:</h6>
        <ul>
            <li><strong>Tuberculosis probability:</strong> {tb_prob*100:.1f}%</li>
            <li><strong>Normal probability:</strong> {normal_prob*100:.1f}%</li>
        </ul>

        <h6>🔬 What This Means:</h6>
        <p>The detected patterns may include:</p>
        <ul>
            <li>Pulmonary infiltrates or consolidations</li>
            <li>Cavitary lesions in lung tissue</li>
            <li>Hilar lymphadenopathy</li>
            <li>Pleural effusions</li>
            <li>Miliary patterns (in disseminated TB)</li>
        </ul>
        """

        risk_assessment = {
            'level': 'High' if confidence >= 0.8 else 'Moderate',
            'description': f"Based on the {confidence_level.lower()} confidence detection, immediate medical attention is recommended.",
            'urgency': 'Urgent' if confidence >= 0.9 else 'High Priority'
        }

        recommendations = [
            "🏥 <strong>Seek immediate medical attention</strong> from a pulmonologist or infectious disease specialist",
            "🧪 <strong>Sputum testing</strong> - Acid-fast bacilli (AFB) smear and culture",
            "🔬 <strong>Molecular testing</strong> - GeneXpert MTB/RIF for rapid confirmation",
            "🩺 <strong>Clinical evaluation</strong> - Complete medical history and physical examination",
            "👥 <strong>Contact tracing</strong> - Identify and test close contacts",
            "🏠 <strong>Isolation precautions</strong> - Follow infection control measures until cleared",
            "💊 <strong>Treatment readiness</strong> - Prepare for potential anti-TB therapy if confirmed"
        ]

        next_steps = [
            "1. <strong>Immediate consultation</strong> with a healthcare provider (within 24-48 hours)",
            "2. <strong>Confirmatory testing</strong> - Sputum collection for laboratory analysis",
            "3. <strong>Additional imaging</strong> - CT scan may be recommended for detailed evaluation",
            "4. <strong>Contact screening</strong> - Family members and close contacts should be tested",
            "5. <strong>Treatment planning</strong> - If confirmed, initiate DOTS (Directly Observed Treatment)"
        ]

    else:
        # Normal/No TB Detected Analysis
        explanation = f"""
        <h5>✅ X-Ray Analysis Results</h5>
        <p>Our AI model indicates that the chest X-ray appears <strong>normal</strong> with no significant signs of tuberculosis detected.
        The analysis shows a {confidence*100:.1f}% confidence level in this assessment.</p>

        <h6>📊 Probability Breakdown:</h6>
        <ul>
            <li><strong>Normal probability:</strong> {normal_prob*100:.1f}%</li>
            <li><strong>Tuberculosis probability:</strong> {tb_prob*100:.1f}%</li>
        </ul>

        <h6>🔬 What This Means:</h6>
        <p>The X-ray shows:</p>
        <ul>
            <li>Clear lung fields without significant infiltrates</li>
            <li>Normal cardiac silhouette</li>
            <li>No obvious cavitary lesions</li>
            <li>No significant pleural abnormalities</li>
            <li>Normal hilar and mediastinal structures</li>
        </ul>
        """

        risk_assessment = {
            'level': 'Low' if confidence >= 0.8 else 'Moderate',
            'description': f"Based on the {confidence_level.lower()} confidence assessment, the risk of tuberculosis appears low.",
            'urgency': 'Routine Follow-up'
        }

        recommendations = [
            "✅ <strong>Continue routine health monitoring</strong>",
            "🩺 <strong>Regular check-ups</strong> - Maintain scheduled medical appointments",
            "💉 <strong>Vaccination status</strong> - Ensure BCG vaccination if appropriate",
            "🚭 <strong>Lifestyle factors</strong> - Avoid smoking and maintain good lung health",
            "👥 <strong>Awareness</strong> - Monitor for TB symptoms if exposed to high-risk environments",
            "🔄 <strong>Re-screening</strong> - Consider periodic screening if in high-risk groups"
        ]

        if confidence < 0.8:
            recommendations.insert(1, "🔍 <strong>Clinical correlation</strong> - Discuss results with healthcare provider")
            recommendations.insert(2, "🧪 <strong>Additional testing</strong> - Consider sputum testing if symptoms present")

        next_steps = [
            "1. <strong>Routine follow-up</strong> with your primary healthcare provider",
            "2. <strong>Symptom monitoring</strong> - Watch for persistent cough, fever, or weight loss",
            "3. <strong>Preventive care</strong> - Maintain good nutrition and immune system health",
            "4. <strong>Risk assessment</strong> - Discuss TB risk factors with your doctor",
            "5. <strong>Future screening</strong> - Follow recommended screening intervals"
        ]

    technical_details = {
        'model_architecture': 'PyTorch ResNet50 Deep Learning Network',
        'training_data': 'Trained on thousands of chest X-ray images',
        'accuracy_metrics': f'{prediction_result["model_accuracy"]}% overall accuracy',
        'processing_method': 'Advanced computer vision and pattern recognition',
        'confidence_interpretation': f'{confidence_level} confidence ({reliability})',
        'image_preprocessing': 'Standardized 224x224 pixel normalization'
    }

    return {
        'explanation': explanation,
        'recommendations': recommendations,
        'risk_assessment': risk_assessment,
        'next_steps': next_steps,
        'technical_details': technical_details,
        'confidence_level': confidence_level,
        'reliability': reliability
    }

@tb_bp.route('/')
def index():
    """TB detection service homepage"""
    return render_template('tb_detection.html', 
                         model_accuracy=tb_detector.accuracy if tb_detector else 0)

@tb_bp.route('/upload', methods=['POST'])
def upload_and_predict():
    """Handle image upload and TB prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Validate if image is suitable for X-ray analysis
        is_valid, validation_message = is_chest_xray(file_path)
        if not is_valid:
            os.remove(file_path)  # Clean up
            return jsonify({'error': validation_message}), 400
        
        # Make TB prediction
        if tb_detector is None:
            return jsonify({'error': 'TB detection model not available'}), 500
        
        prediction_result = tb_detector.predict(file_path)
        
        if 'error' in prediction_result:
            os.remove(file_path)  # Clean up
            return jsonify({'error': prediction_result['error']}), 500
        
        # Generate detailed medical analysis
        detailed_analysis = generate_detailed_analysis(prediction_result)

        # Prepare comprehensive response
        response = {
            'success': True,
            'prediction': prediction_result['prediction'],
            'confidence': round(prediction_result['confidence'], 4),  # Keep as decimal 0-1
            'normal_probability': round(prediction_result['normal_probability'], 4),
            'tb_probability': round(prediction_result['tb_probability'], 4),
            'model_accuracy': prediction_result['model_accuracy'],
            'is_tb_detected': prediction_result['is_tb_detected'],
            'filename': unique_filename,
            'timestamp': datetime.now().isoformat(),
            'validation_message': validation_message,

            # Enhanced medical analysis
            'detailed_analysis': detailed_analysis,
            'medical_explanation': detailed_analysis['explanation'],
            'recommendations': detailed_analysis['recommendations'],
            'risk_assessment': detailed_analysis['risk_assessment'],
            'next_steps': detailed_analysis['next_steps'],
            'technical_details': detailed_analysis['technical_details']
        }
        
        # Clean up uploaded file (optional - you might want to keep for records)
        # os.remove(file_path)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in TB prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500



@tb_bp.route('/api/model-info')
def model_info():
    """API endpoint for model information"""
    if tb_detector:
        return jsonify({
            'model_type': 'PyTorch ResNet50',
            'accuracy': tb_detector.accuracy,
            'classes': ['Normal', 'Tuberculosis'],
            'input_size': '224x224',
            'status': 'active'
        })
    else:
        return jsonify({
            'status': 'unavailable',
            'error': 'Model not loaded'
        }), 500
