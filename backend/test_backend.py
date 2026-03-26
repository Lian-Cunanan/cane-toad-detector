#!/usr/bin/env python3
# =============================================================================
# TEST BACKEND — Quick Start Script
# =============================================================================
#
# This script starts the Flask backend and simulates detection events for testing.
# Use this to test the mobile app without running the actual detector.
#
# Usage:
#   python test_backend.py
# =============================================================================

import sys
import time
import threading
import random
from datetime import datetime

# Add backend to path
sys.path.insert(0, 'backend')

from app import app, socketio, db
from database import SystemStatus, Detection, OperationalPhase
from camera_stream import CameraStream

def simulate_detections():
    """Simulate random detection events for testing."""
    time.sleep(5)  # Wait for server to start
    
    phases = ['Capturing', 'Euthanizing', 'Disposing', 'Heat Sealing']
    current_phase_index = 0
    
    print("\n[SIMULATOR] Starting detection simulation...")
    
    with app.app_context():
        while True:
            try:
                # Random detection every 5-15 seconds
                time.sleep(random.uniform(5, 15))
                
                status = SystemStatus.get_current()
                
                # Create detection
                detection = Detection(
                    confidence=random.uniform(0.7, 0.99),
                    bbox_x1=random.randint(100, 300),
                    bbox_y1=random.randint(100, 300),
                    bbox_x2=random.randint(350, 500),
                    bbox_y2=random.randint(350, 500),
                    class_name='cane_toad'
                )
                db.session.add(detection)
                
                # Update status
                status.current_batch_count += 1
                status.total_captured_today += 1
                status.last_detection_time = datetime.utcnow()
                
                # Random battery drain
                if status.battery_level > 0:
                    status.battery_level = max(0, status.battery_level - random.randint(0, 2))
                
                db.session.commit()
                
                print(f"[SIMULATOR] Detection #{status.current_batch_count}/{status.target_batch_size} | Confidence: {detection.confidence:.2f}")
                
                # Broadcast detection alert
                socketio.emit('detection_alert', {
                    'count': status.current_batch_count,
                    'target': status.target_batch_size,
                    'confidence': detection.confidence,
                    'timestamp': detection.timestamp.isoformat()
                }, namespace='/ws')
                
                # Check if batch is complete
                if status.current_batch_count >= status.target_batch_size:
                    print(f"[SIMULATOR] Batch complete! Moving to next phase...")
                    
                    # Move to next phase
                    current_phase = OperationalPhase.get_current()
                    if current_phase:
                        current_phase.end_phase()
                    
                    current_phase_index = (current_phase_index + 1) % len(phases)
                    new_phase = OperationalPhase(
                        phase_name=phases[current_phase_index],
                        status='active'
                    )
                    db.session.add(new_phase)
                    db.session.commit()
                    
                    # Broadcast phase update
                    socketio.emit('phase_update', new_phase.to_dict(), namespace='/ws')
                    
                    # Reset batch after phase completes
                    time.sleep(10)  # Simulate phase duration
                    status.current_batch_count = 0
                    status.last_batch_reset = datetime.utcnow()
                    db.session.commit()
                    
                    print(f"[SIMULATOR] Phase '{phases[current_phase_index]}' complete. Batch reset.")
                    socketio.emit('batch_reset', {'timestamp': datetime.utcnow().isoformat()}, namespace='/ws')
                    
            except Exception as e:
                print(f"[SIMULATOR ERROR] {e}")
                time.sleep(1)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CANE TOAD DETECTOR - TEST BACKEND")
    print("="*70)
    print("\nThis will start the backend with simulated detections.")
    print("Use this to test the mobile app without hardware.\n")
    print("Backend will be available at: http://0.0.0.0:5000/api/")
    print("WebSocket endpoint: ws://0.0.0.0:5000/ws\n")
    print("Simulated features:")
    print("  [OK] Random detections every 5-15 seconds")
    print("  [OK] Automatic batch completion")
    print("  [OK] Phase transitions (Capturing -> Euthanizing -> etc.)")
    print("  [OK] Battery drain simulation")
    print("  [OK] Live camera feeds (if cameras available)")
    print("\nPress Ctrl+C to stop.\n")
    print("="*70 + "\n")
    
    # Initialize cameras (optional - will fail gracefully if no cameras)
    print("[INFO] Initializing camera streams...")
    try:
        # Import app module to access global camera variables
        import app as app_module
        
        # Try to start cage camera (camera 0)
        try:
            cage_cam = CameraStream(camera_index=0, name="Cage Camera", width=640, height=480, fps=15)
            cage_cam.start()
            if cage_cam.is_active():
                app_module.cage_camera = cage_cam
                print("  [OK] Cage camera started (index 0)")
            else:
                print("  [ERR] Cage camera not available (index 0)")
        except Exception as e:
            print(f"  [ERR] Cage camera failed: {e}")
        
        # Try to start trap camera (camera 1, or use camera 0 if only one available)
        try:
            trap_cam = CameraStream(camera_index=1, name="Trap Camera", width=640, height=480, fps=15)
            trap_cam.start()
            if trap_cam.is_active():
                app_module.trap_camera = trap_cam
                print("  [OK] Trap camera started (index 1)")
            else:
                # Try using same camera as cage for demo purposes
                if app_module.cage_camera:
                    app_module.trap_camera = app_module.cage_camera
                    print("  [INFO] Trap camera using same feed as cage (demo mode)")
                else:
                    print("  [ERR] Trap camera not available (index 1)")
        except Exception as e:
            print(f"  [ERR] Trap camera failed: {e}")
            # Use cage camera for both if available
            if app_module.cage_camera:
                app_module.trap_camera = app_module.cage_camera
                print("  [INFO] Trap camera using same feed as cage (demo mode)")
    
    except Exception as e:
        print(f"[WARN] Camera initialization failed: {e}")
        print("[INFO] Continuing without camera feeds...")
    
    print()
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_detections, daemon=True)
    sim_thread.start()
    
    # Start Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
