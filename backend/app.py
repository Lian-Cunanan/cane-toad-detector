# =============================================================================
# CANE TOAD DETECTOR — Flask Backend API
# =============================================================================
#
# This Flask backend provides RESTful API endpoints and WebSocket support
# for the Kivy mobile application to monitor and control the detector system.
#
# Features:
#   - User authentication (login/registration)
#   - Real-time detection alerts via WebSocket
#   - Detection history and statistics
#   - Camera streaming (MJPEG)
#   - ESP8266 sensor data endpoints
#   - Operational phase tracking (Capturing, Euthanizing, Disposing, Sealing)
#   - System status monitoring
#
# Install dependencies:
#   pip install flask flask-socketio flask-jwt-extended flask-cors python-socketio
#
# Run:
#   python app.py
# =============================================================================

import os
import json
import time
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit, disconnect
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, get_jwt
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from database import db, init_db, User, Detection, SystemStatus, OperationalPhase
from camera_stream import CameraStream

# =============================================================================
# Flask App Configuration
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-key-change-this-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['DATABASE_PATH'] = 'cane_toad_detector.db'

# Enable CORS for mobile app access
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize extensions
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize database
init_db(app)

# Camera streams (will be initialized when detector starts)
cage_camera = None
trap_camera = None

# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400
    
    username = data['username']
    password = data['password']
    
    # Check if user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 409
    
    # Create new user
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token."""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400
    
    username = data['username']
    password = data['password']
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=username)
    
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'username': username
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (client should discard token)."""
    return jsonify({'message': 'Logout successful'}), 200


# =============================================================================
# Detection Endpoints
# =============================================================================

@app.route('/api/detections/current', methods=['GET'])
@jwt_required()
def get_current_detections():
    """Get current detection count and status for active batch."""
    status = SystemStatus.get_current()
    phase = OperationalPhase.get_current()
    
    return jsonify({
        'current_count': status.current_batch_count,
        'target_count': status.target_batch_size,
        'total_captured': status.total_captured_today,
        'phase': phase.phase_name if phase else 'Idle',
        'phase_status': phase.status if phase else 'idle',
        'last_detection': status.last_detection_time.isoformat() if status.last_detection_time else None
    }), 200


@app.route('/api/detections/history', methods=['GET'])
@jwt_required()
def get_detection_history():
    """Get detection history with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    detections = Detection.query.order_by(Detection.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'detections': [d.to_dict() for d in detections.items],
        'total': detections.total,
        'page': page,
        'pages': detections.pages
    }), 200


@app.route('/api/detections/stats', methods=['GET'])
@jwt_required()
def get_detection_stats():
    """Get detection statistics."""
    status = SystemStatus.get_current()
    
    # Get today's detections
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_detections = Detection.query.filter(Detection.timestamp >= today_start).all()
    
    # Get hourly breakdown
    hourly_counts = {}
    for detection in today_detections:
        hour = detection.timestamp.hour
        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
    
    return jsonify({
        'today_total': len(today_detections),
        'current_batch': status.current_batch_count,
        'target_batch': status.target_batch_size,
        'hourly_breakdown': hourly_counts,
        'average_confidence': sum(d.confidence for d in today_detections) / len(today_detections) if today_detections else 0
    }), 200


# =============================================================================
# System Status Endpoints
# =============================================================================

@app.route('/api/status', methods=['GET'])
@jwt_required()
def get_system_status():
    """Get current system status."""
    status = SystemStatus.get_current()
    phase = OperationalPhase.get_current()
    
    return jsonify({
        'battery_level': status.battery_level,
        'wifi_connected': status.wifi_connected,
        'camera_active': status.camera_active,
        'detector_running': status.detector_running,
        'last_update': status.last_update.isoformat(),
        'current_phase': phase.phase_name if phase else 'Idle',
        'phase_status': phase.status if phase else 'idle',
        'current_batch_count': status.current_batch_count,
        'target_batch_size': status.target_batch_size
    }), 200


@app.route('/api/status/update', methods=['POST'])
def update_system_status():
    """Update system status (called by ESP8266 or detector)."""
    # No JWT required - called by hardware
    data = request.get_json()
    
    status = SystemStatus.get_current()
    
    if 'battery_level' in data:
        status.battery_level = data['battery_level']
    if 'wifi_connected' in data:
        status.wifi_connected = data['wifi_connected']
    if 'camera_active' in data:
        status.camera_active = data['camera_active']
    if 'detector_running' in data:
        status.detector_running = data['detector_running']
    
    status.last_update = datetime.utcnow()
    db.session.commit()
    
    # Broadcast status update via WebSocket
    socketio.emit('status_update', status.to_dict(), namespace='/ws')
    
    return jsonify({'message': 'Status updated'}), 200


# =============================================================================
# Operational Phase Endpoints
# =============================================================================

@app.route('/api/phase/current', methods=['GET'])
@jwt_required()
def get_current_phase():
    """Get current operational phase."""
    phase = OperationalPhase.get_current()
    
    if not phase:
        return jsonify({'phase': 'Idle', 'status': 'idle'}), 200
    
    return jsonify(phase.to_dict()), 200


@app.route('/api/phase/update', methods=['POST'])
def update_phase():
    """Update operational phase (called by hardware/detector)."""
    data = request.get_json()
    
    if not data or 'phase' not in data:
        return jsonify({'error': 'Phase name required'}), 400
    
    phase_name = data['phase']
    status = data.get('status', 'active')
    
    # Valid phases: Capturing, Euthanizing, Disposing, Heat Sealing
    valid_phases = ['Capturing', 'Euthanizing', 'Disposing', 'Heat Sealing', 'Idle']
    if phase_name not in valid_phases:
        return jsonify({'error': 'Invalid phase name'}), 400
    
    # End current phase
    current_phase = OperationalPhase.get_current()
    if current_phase:
        current_phase.end_phase()
    
    # Start new phase
    if phase_name != 'Idle':
        new_phase = OperationalPhase(phase_name=phase_name, status=status)
        db.session.add(new_phase)
        db.session.commit()
        
        # Broadcast phase change
        socketio.emit('phase_update', new_phase.to_dict(), namespace='/ws')
    
    return jsonify({'message': 'Phase updated'}), 200


# =============================================================================
# Batch Control Endpoints
# =============================================================================

@app.route('/api/batch/reset', methods=['POST'])
@jwt_required()
def reset_batch():
    """Reset current batch to start new capture cycle."""
    status = SystemStatus.get_current()
    status.current_batch_count = 0
    status.last_batch_reset = datetime.utcnow()
    db.session.commit()
    
    # End current phase
    current_phase = OperationalPhase.get_current()
    if current_phase:
        current_phase.end_phase()
    
    # Broadcast reset event
    socketio.emit('batch_reset', {'timestamp': datetime.utcnow().isoformat()}, namespace='/ws')
    
    return jsonify({'message': 'Batch reset successfully'}), 200


@app.route('/api/batch/settings', methods=['GET', 'POST'])
@jwt_required()
def batch_settings():
    """Get or update batch settings."""
    status = SystemStatus.get_current()
    
    if request.method == 'POST':
        data = request.get_json()
        if 'target_batch_size' in data:
            status.target_batch_size = data['target_batch_size']
            db.session.commit()
        return jsonify({'message': 'Settings updated'}), 200
    
    return jsonify({'target_batch_size': status.target_batch_size}), 200


# =============================================================================
# ESP8266 Endpoints
# =============================================================================

@app.route('/api/esp8266/detection', methods=['POST'])
def esp8266_detection():
    """Receive detection event from ESP8266."""
    data = request.get_json()
    
    # Create detection record
    detection = Detection(
        confidence=data.get('confidence', 0.0),
        bbox_x1=data.get('bbox', [0, 0, 0, 0])[0],
        bbox_y1=data.get('bbox', [0, 0, 0, 0])[1],
        bbox_x2=data.get('bbox', [0, 0, 0, 0])[2],
        bbox_y2=data.get('bbox', [0, 0, 0, 0])[3],
        class_name=data.get('class_name', 'cane_toad')
    )
    db.session.add(detection)
    
    # Update system status
    status = SystemStatus.get_current()
    status.current_batch_count += 1
    status.total_captured_today += 1
    status.last_detection_time = datetime.utcnow()
    db.session.commit()
    
    # Broadcast detection alert
    socketio.emit('detection_alert', {
        'count': status.current_batch_count,
        'target': status.target_batch_size,
        'confidence': detection.confidence,
        'timestamp': detection.timestamp.isoformat()
    }, namespace='/ws')
    
    return jsonify({'message': 'Detection recorded', 'batch_count': status.current_batch_count}), 200


@app.route('/api/esp8266/sensor', methods=['POST'])
def esp8266_sensor_data():
    """Receive sensor data from ESP8266."""
    data = request.get_json()
    
    status = SystemStatus.get_current()
    
    # Update sensor readings
    if 'battery' in data:
        status.battery_level = data['battery']
    if 'wifi_rssi' in data:
        status.wifi_connected = data['wifi_rssi'] > -80  # Good signal threshold
    
    status.last_update = datetime.utcnow()
    db.session.commit()
    
    # Broadcast sensor update
    socketio.emit('sensor_update', data, namespace='/ws')
    
    return jsonify({'message': 'Sensor data received'}), 200


# =============================================================================
# Camera Streaming Endpoints
# =============================================================================

@app.route('/api/camera/cage/stream')
def cage_camera_stream():
    """Stream cage camera feed (MJPEG)."""
    # Accept token as query parameter for compatibility with image widgets
    token = request.args.get('token')
    if not token:
        return jsonify({'error': 'Token required as query parameter'}), 401
    
    # Verify token manually
    try:
        from flask_jwt_extended import decode_token
        decode_token(token)
    except Exception as e:
        return jsonify({'error': 'Invalid token'}), 401
    
    if not cage_camera:
        return jsonify({'error': 'Cage camera not available'}), 503
    
    return Response(
        cage_camera.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/trap/stream')
def trap_camera_stream():
    """Stream trap camera feed (MJPEG)."""
    # Accept token as query parameter for compatibility with image widgets
    token = request.args.get('token')
    if not token:
        return jsonify({'error': 'Token required as query parameter'}), 401
    
    # Verify token manually
    try:
        from flask_jwt_extended import decode_token
        decode_token(token)
    except Exception as e:
        return jsonify({'error': 'Invalid token'}), 401
    
    if not trap_camera:
        return jsonify({'error': 'Trap camera not available'}), 503
    
    return Response(
        trap_camera.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/status')
@jwt_required()
def camera_status():
    """Get camera availability status."""
    return jsonify({
        'cage_camera': cage_camera is not None and cage_camera.is_active(),
        'trap_camera': trap_camera is not None and trap_camera.is_active()
    }), 200


# =============================================================================
# WebSocket Events
# =============================================================================

@socketio.on('connect', namespace='/ws')
def handle_connect():
    """Handle WebSocket connection."""
    print(f"[WebSocket] Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Cane Toad Detector'})


@socketio.on('disconnect', namespace='/ws')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print(f"[WebSocket] Client disconnected: {request.sid}")


@socketio.on('ping', namespace='/ws')
def handle_ping():
    """Handle ping from client to keep connection alive."""
    emit('pong', {'timestamp': datetime.utcnow().isoformat()})


# =============================================================================
# Helper Routes
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/api/alerts/test', methods=['POST'])
@jwt_required()
def test_alert():
    """Test alert system by sending a test notification."""
    socketio.emit('test_alert', {
        'message': 'This is a test alert',
        'timestamp': datetime.utcnow().isoformat()
    }, namespace='/ws')
    
    return jsonify({'message': 'Test alert sent'}), 200


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("CANE TOAD DETECTOR — Flask Backend API")
    print("="*70)
    print(f"\nBackend server starting...")
    print(f"API will be available at: http://0.0.0.0:5000/api/")
    print(f"WebSocket endpoint: ws://0.0.0.0:5000/ws")
    print("\nAvailable endpoints:")
    print("  - POST /api/auth/register")
    print("  - POST /api/auth/login")
    print("  - GET  /api/detections/current")
    print("  - GET  /api/status")
    print("  - POST /api/batch/reset")
    print("  - GET  /api/camera/cage/stream")
    print("\nPress Ctrl+C to stop.\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
