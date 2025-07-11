#!/usr/bin/env python3
"""
AI Health Assistant Service
===========================

A comprehensive AI-powered health assistant that provides:
- Medical Q&A with AI models
- Symptom analysis and guidance
- Medication information
- Health education
- Emergency guidance

Supports multiple AI providers: OpenAI, Anthropic, Google, etc.
"""

import os
import json
import logging
import requests
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class AIHealthAssistant:
    """Main AI Health Assistant class"""
    
    def __init__(self):
        self.api_key = None
        self.api_provider = "google"  # Default to Google Gemini
        self.model_name = "gemini-pro"
        self.conversation_history = []
        self.max_history = 10

        # Load API configuration
        self.load_api_config()
        
        # Health-specific prompts and guidelines
        self.system_prompt = """You are a helpful AI health assistant. You provide general health information, wellness guidance, and educational content. 

IMPORTANT GUIDELINES:
1. Always emphasize that you are not a replacement for professional medical advice
2. Encourage users to consult healthcare providers for serious concerns
3. Provide general, educational information only
4. Do not diagnose medical conditions
5. For emergencies, direct users to call emergency services
6. Be empathetic and supportive
7. Use clear, understandable language
8. Provide evidence-based information when possible

You can help with:
- General health questions
- Wellness tips
- Basic symptom information (educational only)
- Medication general information
- Healthy lifestyle guidance
- Mental health support resources
- Preventive care information

Always end responses with appropriate disclaimers when discussing health topics."""

    def load_api_config(self):
        """Load API configuration from environment or config file"""
        # Try to load from environment variables
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('AI_API_KEY')

        # Try to load from config file (check multiple possible locations)
        config_paths = ['ai_config.json', '../ai_config.json', '../../ai_config.json']

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.api_key = config.get('api_key')
                        self.api_provider = config.get('provider', 'google')
                        self.model_name = config.get('model', 'gemini-1.5-flash')
                        logger.info(f"✅ AI config loaded from {config_path}: {self.api_provider} - {self.model_name}")
                        break
                except Exception as e:
                    logger.error(f"Error loading AI config from {config_path}: {e}")
                    continue

    def set_api_key(self, api_key: str, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """Set API key and provider"""
        self.api_key = api_key
        self.api_provider = provider
        self.model_name = model
        
        # Save to config file
        config = {
            'api_key': api_key,
            'provider': provider,
            'model': model
        }
        try:
            # Try to save in the main directory
            config_path = 'ai_config.json'
            if not os.path.exists(config_path):
                # If we're in a subdirectory, try parent directories
                for path in ['../ai_config.json', '../../ai_config.json']:
                    if os.path.exists(os.path.dirname(path) if os.path.dirname(path) else '.'):
                        config_path = path
                        break

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"AI configuration saved to {config_path} for provider: {provider}")
        except Exception as e:
            logger.error(f"Error saving AI config: {e}")

    def is_configured(self) -> bool:
        """Check if AI assistant is properly configured"""
        return self.api_key is not None

    def add_to_history(self, user_message: str, ai_response: str):
        """Add conversation to history"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': ai_response
        })
        
        # Keep only recent conversations
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_openai_response(self, user_message: str) -> str:
        """Get response from OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Build conversation context
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent conversation history
            for conv in self.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": conv['user']})
                messages.append({"role": "assistant", "content": conv['assistant']})
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later."
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "I'm experiencing technical difficulties. Please try again later."

    def get_google_response(self, user_message: str) -> str:
        """Get response from Google Gemini API"""
        try:
            headers = {
                'Content-Type': 'application/json'
            }

            # Build conversation context for Gemini
            conversation_context = self.system_prompt + "\n\n"

            # Add recent conversation history
            for conv in self.conversation_history[-3:]:  # Last 3 exchanges
                conversation_context += f"User: {conv['user']}\nAssistant: {conv['assistant']}\n\n"

            # Add current message
            conversation_context += f"User: {user_message}\nAssistant:"

            data = {
                "contents": [{
                    "parts": [{
                        "text": conversation_context
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 500,
                }
            }

            url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}'

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                else:
                    logger.error(f"No content in Gemini response: {result}")
                    return "I'm sorry, I couldn't generate a response. Please try again."
            else:
                logger.error(f"Google Gemini API error: {response.status_code} - {response.text}")
                return "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later."

        except Exception as e:
            logger.error(f"Error calling Google Gemini API: {e}")
            return "I'm experiencing technical difficulties. Please try again later."

    def get_fallback_response(self, user_message: str) -> str:
        """Provide fallback response when AI API is not available"""
        user_lower = user_message.lower()
        
        # Emergency keywords
        emergency_keywords = ['emergency', 'urgent', 'chest pain', 'heart attack', 'stroke', 'bleeding', 'unconscious']
        if any(keyword in user_lower for keyword in emergency_keywords):
            return """🚨 EMERGENCY RESPONSE:
If this is a medical emergency, please:
- Call 911 (US) or your local emergency number immediately
- Go to the nearest emergency room
- Contact emergency medical services

For chest pain, difficulty breathing, severe bleeding, or loss of consciousness, seek immediate medical attention.

This AI assistant cannot handle medical emergencies. Please contact emergency services right away."""

        # Common health topics
        if 'symptom' in user_lower:
            return """I can provide general information about symptoms, but I cannot diagnose medical conditions. 

For any concerning symptoms:
1. Consult with a healthcare provider
2. Keep track of when symptoms occur
3. Note any triggers or patterns
4. Seek immediate care for severe symptoms

Would you like general information about maintaining good health or when to seek medical care?

⚠️ Disclaimer: This is not medical advice. Always consult healthcare professionals for medical concerns."""

        if 'medication' in user_lower or 'medicine' in user_lower:
            return """For medication questions:
1. Always consult your pharmacist or doctor
2. Read medication labels carefully
3. Follow prescribed dosages exactly
4. Report side effects to your healthcare provider
5. Don't stop medications without medical guidance

For specific medication information, please speak with your pharmacist or healthcare provider.

⚠️ Disclaimer: This is general information only, not medical advice."""

        # General health response
        return """I'm here to help with general health information and wellness guidance. I can discuss:

🏥 General health topics
💊 Basic medication information (not specific advice)
🏃‍♀️ Wellness and lifestyle tips
🧠 Mental health resources
🍎 Nutrition basics
💤 Sleep hygiene

Please remember that I cannot:
- Diagnose medical conditions
- Provide specific medical advice
- Replace professional healthcare

What health topic would you like to learn about?

⚠️ Always consult healthcare professionals for medical advice and treatment."""

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process user message and return AI response"""
        try:
            # Check for emergency keywords first
            emergency_keywords = ['emergency', 'urgent', 'chest pain', 'heart attack', 'stroke', 'bleeding', 'unconscious', '911']
            if any(keyword in user_message.lower() for keyword in emergency_keywords):
                response = """🚨 MEDICAL EMERGENCY DETECTED

If this is a medical emergency:
• Call 911 immediately (US) or your local emergency number
• Go to the nearest emergency room
• Contact emergency medical services

For life-threatening symptoms like chest pain, difficulty breathing, severe bleeding, or loss of consciousness, seek immediate medical attention.

This AI assistant cannot handle medical emergencies. Please contact emergency services right away."""
                
                return {
                    'response': response,
                    'type': 'emergency',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'emergency_protocol'
                }

            # Try AI API if configured
            if self.is_configured():
                if self.api_provider == "openai":
                    ai_response = self.get_openai_response(user_message)
                elif self.api_provider == "google":
                    ai_response = self.get_google_response(user_message)
                else:
                    ai_response = self.get_fallback_response(user_message)

                source = f"{self.api_provider}_api"
            else:
                ai_response = self.get_fallback_response(user_message)
                source = "fallback_system"

            # Add to conversation history
            self.add_to_history(user_message, ai_response)
            
            return {
                'response': ai_response,
                'type': 'health_guidance',
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'configured': self.is_configured()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I'm sorry, I encountered an error while processing your message. Please try again.",
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'source': 'error_handler'
            }

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Global AI assistant instance
ai_assistant = AIHealthAssistant()

# Blueprint for AI Health Assistant
ai_health_bp = Blueprint('ai_health', __name__, 
                        template_folder='templates',
                        static_folder='static')

@ai_health_bp.route('/')
def index():
    """AI Health Assistant homepage"""
    return render_template('ai_assistant.html')

@ai_health_bp.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process message with AI assistant
        result = ai_assistant.process_message(user_message)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@ai_health_bp.route('/api/configure', methods=['POST'])
def configure():
    """Configure AI API settings"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-3.5-turbo')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        # Set configuration
        ai_assistant.set_api_key(api_key, provider, model)
        
        return jsonify({
            'success': True,
            'message': f'AI assistant configured with {provider}',
            'provider': provider,
            'model': model
        })
        
    except Exception as e:
        logger.error(f"Error configuring AI: {e}")
        return jsonify({'error': 'Configuration failed'}), 500

@ai_health_bp.route('/api/history')
def get_history():
    """Get conversation history"""
    try:
        history = ai_assistant.get_conversation_history()
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': 'Failed to get history'}), 500

@ai_health_bp.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        ai_assistant.clear_history()
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500
