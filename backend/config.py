import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///data/healthcare_portal.db'
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # AI Configuration
    AI_CONFIG_PATH = 'services/ai_assistant/config.json'
    
    # Model paths
    TB_MODEL_PATH = 'models/pytorch_tb_model.pth'
    
    # CORS settings
    CORS_ORIGINS = ['*']  # Allow all origins for development

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}