# =============================================================================
# CAMERA STREAMING — MJPEG Stream for Mobile App
# =============================================================================
#
# Provides MJPEG streaming for camera feeds to be consumed by mobile app.
# Supports multiple camera sources (cage view and trap view).
# =============================================================================

import cv2
import threading
import time
from typing import Optional


class CameraStream:
    """
    MJPEG camera stream handler for Flask.
    Captures frames from a camera and serves them as MJPEG stream.
    """
    
    def __init__(self, camera_index: int = 0, name: str = "Camera", 
                 width: int = 640, height: int = 480, fps: int = 15):
        """
        Initialize camera stream.
        
        Args:
            camera_index: Camera device index (0 for built-in, 1+ for USB)
            name: Descriptive name for this camera
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second for streaming
        """
        self.camera_index = camera_index
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        self.capture = None
        self.current_frame = None
        self.lock = threading.Lock()
        self.active = False
        self.thread = None
        
    def start(self):
        """Start the camera stream."""
        if self.active:
            return
        
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                print(f"[ERROR] Could not open camera {self.camera_index} ({self.name})")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.active = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            print(f"[INFO] Camera stream started: {self.name} (index {self.camera_index})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start camera {self.name}: {e}")
            return False
    
    def stop(self):
        """Stop the camera stream."""
        self.active = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        print(f"[INFO] Camera stream stopped: {self.name}")
    
    def _capture_loop(self):
        """Continuous capture loop running in background thread."""
        while self.active:
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    print(f"[WARN] Failed to read frame from {self.name}")
                    time.sleep(0.1)
                    continue
                
                # Update current frame
                with self.lock:
                    self.current_frame = frame.copy()
                
                # Maintain target FPS
                time.sleep(self.frame_delay)
                
            except Exception as e:
                print(f"[ERROR] Camera capture error ({self.name}): {e}")
                time.sleep(0.5)
    
    def get_frame(self) -> Optional[bytes]:
        """
        Get the current frame as JPEG bytes.
        
        Returns:
            JPEG-encoded frame as bytes, or None if no frame available
        """
        with self.lock:
            if self.current_frame is None:
                return None
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', self.current_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret:
                return None
            
            return jpeg.tobytes()
    
    def generate_frames(self):
        """
        Generator function for MJPEG streaming.
        Yields frames in multipart format for HTTP streaming.
        """
        while True:
            frame_bytes = self.get_frame()
            
            if frame_bytes is None:
                # No frame available, wait and retry
                time.sleep(0.1)
                continue
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control streaming rate
            time.sleep(self.frame_delay)
    
    def is_active(self) -> bool:
        """Check if camera stream is active."""
        return self.active and self.capture is not None and self.capture.isOpened()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


# =============================================================================
# Camera Manager
# =============================================================================

class CameraManager:
    """Manage multiple camera streams."""
    
    def __init__(self):
        self.cameras = {}
    
    def add_camera(self, name: str, camera_index: int, width: int = 640, 
                   height: int = 480, fps: int = 15) -> CameraStream:
        """Add a camera stream."""
        camera = CameraStream(camera_index, name, width, height, fps)
        self.cameras[name] = camera
        return camera
    
    def start_camera(self, name: str) -> bool:
        """Start a specific camera."""
        if name not in self.cameras:
            return False
        return self.cameras[name].start()
    
    def stop_camera(self, name: str):
        """Stop a specific camera."""
        if name in self.cameras:
            self.cameras[name].stop()
    
    def start_all(self):
        """Start all cameras."""
        for camera in self.cameras.values():
            camera.start()
    
    def stop_all(self):
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()
    
    def get_camera(self, name: str) -> Optional[CameraStream]:
        """Get a camera by name."""
        return self.cameras.get(name)
    
    def is_active(self, name: str) -> bool:
        """Check if a camera is active."""
        camera = self.cameras.get(name)
        return camera.is_active() if camera else False


# Example usage:
if __name__ == "__main__":
    print("Testing camera stream...")
    
    # Create camera stream
    camera = CameraStream(camera_index=0, name="Test Camera")
    
    # Start streaming
    if camera.start():
        print("Camera started. Press Ctrl+C to stop.")
        
        try:
            while True:
                frame = camera.get_frame()
                if frame:
                    print(f"Frame captured: {len(frame)} bytes")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
    
    camera.stop()
    print("Done.")
