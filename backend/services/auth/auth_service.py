#!/usr/bin/env python3
"""
User Authentication Service
===========================

Handles user registration, login, and session management for the TB Detection Platform.

Author: TB Detection Platform Team
Version: 1.0.0
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__)

class AuthService:
    def __init__(self, db_path='data/users.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the users database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not create database directory: {e}")
            # Try to use a temporary directory or current directory
            self.db_path = 'users.db'
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, username, email, password, full_name):
        """Register a new user"""
        try:
            # Validate input
            if not all([username, email, password, full_name]):
                return {'success': False, 'error': 'All fields are required'}
            
            if len(password) < 6:
                return {'success': False, 'error': 'Password must be at least 6 characters'}
            
            # Hash password
            password_hash = generate_password_hash(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'error': 'Username or email already exists'}
            
            # Insert new user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Registration failed: {str(e)}'}
    
    def login_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user by username or email
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, is_active
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return {'success': False, 'error': 'Invalid username or password'}
            
            user_id, db_username, email, password_hash, full_name, is_active = user
            
            # Check password
            if not check_password_hash(password_hash, password):
                conn.close()
                return {'success': False, 'error': 'Invalid username or password'}
            
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user_id))
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=7)  # 7 days session
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': 'Login successful',
                'user': {
                    'id': user_id,
                    'username': db_username,
                    'email': email,
                    'full_name': full_name
                },
                'session_token': session_token
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Login failed: {str(e)}'}
    
    def verify_session(self, session_token):
        """Verify if session token is valid"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.full_name
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.is_active = 1 
                AND s.expires_at > ? AND u.is_active = 1
            ''', (session_token, datetime.now()))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    'success': True,
                    'user': {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'full_name': user[3]
                    }
                }
            else:
                return {'success': False, 'error': 'Invalid or expired session'}
                
        except Exception as e:
            return {'success': False, 'error': f'Session verification failed: {str(e)}'}
    
    def logout_user(self, session_token):
        """Logout user by invalidating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE session_token = ?
            ''', (session_token,))
            
            conn.commit()
            conn.close()
            
            return {'success': True, 'message': 'Logged out successfully'}
            
        except Exception as e:
            return {'success': False, 'error': f'Logout failed: {str(e)}'}

# Initialize auth service
auth_service = AuthService()

# API Routes
@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    result = auth_service.register_user(
        data.get('username'),
        data.get('email'),
        data.get('password'),
        data.get('full_name')
    )
    
    status_code = 200 if result['success'] else 400
    return jsonify(result), status_code

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    result = auth_service.login_user(
        data.get('username'),
        data.get('password')
    )
    
    status_code = 200 if result['success'] else 401
    return jsonify(result), status_code

@auth_bp.route('/verify', methods=['POST'])
def verify():
    """Verify session token"""
    data = request.get_json()
    
    if not data or not data.get('session_token'):
        return jsonify({'success': False, 'error': 'Session token required'}), 400
    
    result = auth_service.verify_session(data.get('session_token'))
    
    status_code = 200 if result['success'] else 401
    return jsonify(result), status_code

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    data = request.get_json()
    
    if not data or not data.get('session_token'):
        return jsonify({'success': False, 'error': 'Session token required'}), 400
    
    result = auth_service.logout_user(data.get('session_token'))
    
    return jsonify(result), 200
