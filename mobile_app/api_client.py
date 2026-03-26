# =============================================================================
# API CLIENT — Backend Communication
# =============================================================================
#
# Handles all HTTP requests and WebSocket communication with Flask backend.
# =============================================================================

import requests
import socketio
import threading
from typing import Callable, Optional, Dict, Any


class APIClient:
    """
    API client for communicating with Flask backend.
    Handles HTTP requests and WebSocket connections.
    """
    
    def __init__(self, base_url: str):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the backend server (e.g., 'http://192.168.1.100:5000')
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.token: Optional[str] = None
        self.sio: Optional[socketio.Client] = None
        self.connected = False
        
        # Callbacks for WebSocket events
        self.on_detection = None
        self.on_status_update = None
        self.on_phase_update = None
        self.on_alert = None
        
    def set_token(self, token: str):
        """Set JWT authentication token."""
        self.token = token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers
    
    def _make_request(self, method: str, endpoint: str, 
                     callback: Optional[Callable] = None, **kwargs):
        """
        Make HTTP request to backend.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            callback: Callback function(success: bool, data: dict)
            **kwargs: Additional arguments for requests
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        def request_thread():
            try:
                response = requests.request(
                    method, 
                    url, 
                    headers=self._get_headers(),
                    timeout=10,
                    **kwargs
                )
                
                success = response.status_code < 400
                data = response.json() if response.content else {}
                
                if callback:
                    callback(success, data)
                    
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {e}")
                if callback:
                    callback(False, {'error': str(e)})
        
        # Run request in background thread
        thread = threading.Thread(target=request_thread, daemon=True)
        thread.start()
    
    # =========================================================================
    # Authentication Endpoints
    # =========================================================================
    
    def register(self, username: str, password: str, callback: Callable):
        """Register new user account."""
        self._make_request(
            'POST',
            '/auth/register',
            callback=callback,
            json={'username': username, 'password': password}
        )
    
    def login(self, username: str, password: str, callback: Callable):
        """Login user and get JWT token."""
        self._make_request(
            'POST',
            '/auth/login',
            callback=callback,
            json={'username': username, 'password': password}
        )
    
    def logout(self):
        """Logout user."""
        if self.token:
            self._make_request('POST', '/auth/logout')
        self.token = None
    
    # =========================================================================
    # Detection Endpoints
    # =========================================================================
    
    def get_current_detections(self, callback: Callable):
        """Get current detection count and status."""
        self._make_request('GET', '/detections/current', callback=callback)
    
    def get_detection_history(self, page: int = 1, callback: Callable = None):
        """Get detection history with pagination."""
        self._make_request(
            'GET', 
            f'/detections/history?page={page}',
            callback=callback
        )
    
    def get_detection_stats(self, callback: Callable):
        """Get detection statistics."""
        self._make_request('GET', '/detections/stats', callback=callback)
    
    # =========================================================================
    # System Status Endpoints
    # =========================================================================
    
    def get_system_status(self, callback: Callable):
        """Get current system status."""
        self._make_request('GET', '/status', callback=callback)
    
    def update_system_status(self, status_data: Dict[str, Any], callback: Callable = None):
        """Update system status."""
        self._make_request(
            'POST',
            '/status/update',
            callback=callback,
            json=status_data
        )
    
    # =========================================================================
    # Batch Control Endpoints
    # =========================================================================
    
    def reset_batch(self, callback: Callable):
        """Reset current batch count."""
        self._make_request('POST', '/batch/reset', callback=callback)
    
    def get_batch_settings(self, callback: Callable):
        """Get batch settings."""
        self._make_request('GET', '/batch/settings', callback=callback)
    
    def update_batch_settings(self, target_size: int, callback: Callable):
        """Update batch target size."""
        self._make_request(
            'POST',
            '/batch/settings',
            callback=callback,
            json={'target_batch_size': target_size}
        )
    
    # =========================================================================
    # Camera Endpoints
    # =========================================================================
    
    def get_camera_status(self, callback: Callable):
        """Get camera availability status."""
        self._make_request('GET', '/camera/status', callback=callback)
    
    def get_cage_stream_url(self) -> str:
        """Get cage camera stream URL."""
        return f"{self.api_url}/camera/cage/stream"
    
    def get_trap_stream_url(self) -> str:
        """Get trap camera stream URL."""
        return f"{self.api_url}/camera/trap/stream"
    
    # =========================================================================
    # WebSocket Methods
    # =========================================================================
    
    def connect_websocket(self):
        """Connect to WebSocket for real-time updates."""
        if self.connected:
            return
        
        try:
            self.sio = socketio.Client()
            
            # Register event handlers
            @self.sio.on('connect', namespace='/ws')
            def on_connect():
                print("[WebSocket] Connected to backend")
                self.connected = True
            
            @self.sio.on('disconnect', namespace='/ws')
            def on_disconnect():
                print("[WebSocket] Disconnected from backend")
                self.connected = False
            
            @self.sio.on('detection_alert', namespace='/ws')
            def on_detection_alert(data):
                print(f"[WebSocket] Detection alert: {data}")
                if self.on_detection:
                    self.on_detection(data)
            
            @self.sio.on('status_update', namespace='/ws')
            def on_status_update_event(data):
                print(f"[WebSocket] Status update: {data}")
                if self.on_status_update:
                    self.on_status_update(data)
            
            @self.sio.on('phase_update', namespace='/ws')
            def on_phase_update_event(data):
                print(f"[WebSocket] Phase update: {data}")
                if self.on_phase_update:
                    self.on_phase_update(data)
            
            @self.sio.on('test_alert', namespace='/ws')
            def on_test_alert(data):
                print(f"[WebSocket] Test alert: {data}")
                if self.on_alert:
                    self.on_alert(data)
            
            # Connect
            self.sio.connect(self.base_url, namespaces=['/ws'])
            
            return True
            
        except Exception as e:
            print(f"[ERROR] WebSocket connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect WebSocket."""
        if self.sio and self.connected:
            self.sio.disconnect()
            self.connected = False
    
    def set_detection_callback(self, callback: Callable):
        """Set callback for detection alerts."""
        self.on_detection = callback
    
    def set_status_update_callback(self, callback: Callable):
        """Set callback for status updates."""
        self.on_status_update = callback
    
    def set_phase_update_callback(self, callback: Callable):
        """Set callback for phase updates."""
        self.on_phase_update = callback
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for alerts."""
        self.on_alert = callback
