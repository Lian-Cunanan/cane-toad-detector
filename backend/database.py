# =============================================================================
# DATABASE MODELS — Cane Toad Detector Backend
# =============================================================================
#
# SQLAlchemy models for:
#   - User authentication
#   - Detection records
#   - System status
#   - Operational phases
# =============================================================================

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# =============================================================================
# Initialize Database
# =============================================================================

def init_db(app):
    """Initialize database with Flask app."""
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{app.config['DATABASE_PATH']}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Create default system status if it doesn't exist
        if not SystemStatus.query.first():
            default_status = SystemStatus()
            db.session.add(default_status)
            db.session.commit()
            print("[DB] Default system status created")


# =============================================================================
# User Model
# =============================================================================

class User(db.Model):
    """User accounts for mobile app authentication."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


# =============================================================================
# Detection Model
# =============================================================================

class Detection(db.Model):
    """Individual toad detection records."""
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    confidence = db.Column(db.Float, nullable=False)
    class_name = db.Column(db.String(50), default='cane_toad')
    
    # Bounding box coordinates
    bbox_x1 = db.Column(db.Integer)
    bbox_y1 = db.Column(db.Integer)
    bbox_x2 = db.Column(db.Integer)
    bbox_y2 = db.Column(db.Integer)
    
    # Optional: image path if saving detection snapshots
    image_path = db.Column(db.String(255))
    
    # Batch association
    batch_number = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'class_name': self.class_name,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'batch_number': self.batch_number
        }
    
    def __repr__(self):
        return f'<Detection {self.id} at {self.timestamp}>'


# =============================================================================
# System Status Model
# =============================================================================

class SystemStatus(db.Model):
    """Current system status (singleton - only one record)."""
    __tablename__ = 'system_status'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Battery and connectivity
    battery_level = db.Column(db.Integer, default=100)  # 0-100%
    wifi_connected = db.Column(db.Boolean, default=True)
    
    # System state
    camera_active = db.Column(db.Boolean, default=False)
    detector_running = db.Column(db.Boolean, default=False)
    
    # Detection counters
    current_batch_count = db.Column(db.Integer, default=0)
    target_batch_size = db.Column(db.Integer, default=10)
    total_captured_today = db.Column(db.Integer, default=0)
    
    # Timestamps
    last_detection_time = db.Column(db.DateTime)
    last_batch_reset = db.Column(db.DateTime)
    last_update = db.Column(db.DateTime, default=datetime.utcnow)
    
    @staticmethod
    def get_current():
        """Get the current system status (singleton)."""
        status = SystemStatus.query.first()
        if not status:
            status = SystemStatus()
            db.session.add(status)
            db.session.commit()
        return status
    
    def to_dict(self):
        return {
            'battery_level': self.battery_level,
            'wifi_connected': self.wifi_connected,
            'camera_active': self.camera_active,
            'detector_running': self.detector_running,
            'current_batch_count': self.current_batch_count,
            'target_batch_size': self.target_batch_size,
            'total_captured_today': self.total_captured_today,
            'last_detection_time': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'last_batch_reset': self.last_batch_reset.isoformat() if self.last_batch_reset else None,
            'last_update': self.last_update.isoformat()
        }
    
    def __repr__(self):
        return f'<SystemStatus batch:{self.current_batch_count}/{self.target_batch_size}>'


# =============================================================================
# Operational Phase Model
# =============================================================================

class OperationalPhase(db.Model):
    """Track operational phases: Capturing, Euthanizing, Disposing, Heat Sealing."""
    __tablename__ = 'operational_phases'
    
    id = db.Column(db.Integer, primary_key=True)
    phase_name = db.Column(db.String(50), nullable=False)  # Capturing, Euthanizing, etc.
    status = db.Column(db.String(20), default='active')  # active, completed, error
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    
    # Phase-specific data (JSON stored as string)
    phase_data = db.Column(db.Text)  # Can store JSON data
    
    @staticmethod
    def get_current():
        """Get the currently active phase."""
        return OperationalPhase.query.filter_by(status='active').order_by(
            OperationalPhase.start_time.desc()
        ).first()
    
    def end_phase(self):
        """Mark phase as completed."""
        self.end_time = datetime.utcnow()
        self.status = 'completed'
        self.duration_seconds = int((self.end_time - self.start_time).total_seconds())
        db.session.commit()
    
    def to_dict(self):
        return {
            'id': self.id,
            'phase_name': self.phase_name,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds
        }
    
    def __repr__(self):
        return f'<OperationalPhase {self.phase_name} - {self.status}>'


# =============================================================================
# Alert/Notification Model
# =============================================================================

class Alert(db.Model):
    """System alerts and notifications."""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50), nullable=False)  # motion, intruder, low_battery, etc.
    message = db.Column(db.String(255), nullable=False)
    severity = db.Column(db.String(20), default='info')  # info, warning, critical
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'alert_type': self.alert_type,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }
    
    def __repr__(self):
        return f'<Alert {self.alert_type} - {self.severity}>'
