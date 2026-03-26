# =============================================================================
# CANE TOAD DETECTOR — Kivy Mobile Application
# =============================================================================
#
# Cross-platform mobile app for monitoring and controlling the detector system.
#
# Features:
#   - Real-time dashboard with progress tracking
#   - Dual camera feeds (Cage View & Trap View)
#   - Operational phase monitoring
#   - System status display (battery, WiFi, etc.)
#   - Alert notifications
#   - Batch control (reset, next batch)
#
# Install dependencies:
#   pip install kivy kivymd requests python-socketio
#
# Run:
#   python main.py
#
# Build APK for Android:
#   buildozer android debug
# =============================================================================

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

import requests
import socketio
from typing import Optional

from screens.dashboard_screen import DashboardScreen
from screens.camera_screen import CameraScreen
from screens.settings_screen import SettingsScreen
from api_client import APIClient

from kivy.storage.jsonstore import JsonStore
from pathlib import Path


# Use a phone-like viewport when running on desktop for consistent UI testing.
if platform not in ('android', 'ios'):
    Window.size = (360, 740)


# =============================================================================
# Main App Class
# =============================================================================

class CaneToadDetectorApp(MDApp):
    """Main Kivy application class."""
    
    # App properties
    backend_url = StringProperty('http://192.168.4.1:5000')  # Default Pi hotspot IP
    is_connected = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client: Optional[APIClient] = None
        self.screen_manager: Optional[ScreenManager] = None
        self.dialog: Optional[MDDialog] = None
        self.store: Optional[JsonStore] = None
        
    def build(self):
        """Build the app UI."""
        self.title = "Cane Toad Detector"
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.theme_style = "Dark"
        
        # Load saved settings
        self.load_settings()
        
        # Initialize API client
        self.api_client = APIClient(self.backend_url)
        
        # Create screen manager
        self.screen_manager = ScreenManager(transition=SlideTransition())
        
        # Add screens
        self.screen_manager.add_widget(DashboardScreen(name='dashboard'))
        self.screen_manager.add_widget(CameraScreen(name='camera'))
        self.screen_manager.add_widget(SettingsScreen(name='settings'))
        
        return self.screen_manager
    
    def on_start(self):
        """Called when app starts."""
        print(f"[INFO] Cane Toad Detector app started")
        print(f"[INFO] Backend URL: {self.backend_url}")
        
        # Connect to backend immediately (no login required)
        self.api_client.connect_websocket()
        self.is_connected = True
        
    def on_stop(self):
        """Called when app stops."""
        if self.api_client:
            self.api_client.disconnect()
        print("[INFO] App stopped")
    
    # =========================================================================
    # Settings Methods
    # =========================================================================
    
    def load_settings(self):
        """Load saved settings from persistent storage."""
        try:
            storage_path = str(Path.home() / '.cane_toad_detector')
            Path(storage_path).mkdir(exist_ok=True)
            self.store = JsonStore(str(Path(storage_path) / 'settings.json'))
            
            # Load backend URL if saved
            if self.store.exists('backend'):
                saved_url = self.store.get('backend')['url']
                self.backend_url = saved_url
                print(f"[INFO] Loaded saved backend URL: {saved_url}")
            else:
                print(f"[INFO] Using default backend URL: {self.backend_url}")
        except Exception as e:
            print(f"[WARNING] Failed to load settings: {e}")
    
    # =========================================================================
    # Dialog Methods
    # =========================================================================
    
    def show_error(self, message: str):
        """Show error dialog."""
        def create_dialog(dt):
            self.dismiss_dialog()
            self.dialog = MDDialog(
                title="Error",
                text=message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: self.dismiss_dialog()
                    )
                ]
            )
            self.dialog.open()
        Clock.schedule_once(create_dialog, 0)
    
    def show_success(self, message: str):
        """Show success dialog."""
        def create_dialog(dt):
            self.dismiss_dialog()
            self.dialog = MDDialog(
                title="Success",
                text=message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: self.dismiss_dialog()
                    )
                ]
            )
            self.dialog.open()
        Clock.schedule_once(create_dialog, 0)
    
    def show_info(self, message: str):
        """Show info dialog."""
        def create_dialog(dt):
            self.dismiss_dialog()
            self.dialog = MDDialog(
                title="Info",
                text=message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: self.dismiss_dialog()
                    )
                ]
            )
            self.dialog.open()
        Clock.schedule_once(create_dialog, 0)
    
    def show_loading(self, message: str = "Loading..."):
        """Show loading dialog."""
        def create_dialog(dt):
            self.dismiss_dialog()
            self.dialog = MDDialog(
                title="Please wait",
                text=message
            )
            self.dialog.open()
        Clock.schedule_once(create_dialog, 0)
    
    def dismiss_dialog(self):
        """Dismiss current dialog."""
        if self.dialog:
            try:
                self.dialog.dismiss()
            except:
                pass
            self.dialog = None
    
    def show_confirmation(self, title: str, message: str, on_confirm):
        """Show confirmation dialog."""
        def create_dialog(dt):
            self.dismiss_dialog()
            self.dialog = MDDialog(
                title=title,
                text=message,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        on_release=lambda x: self.dismiss_dialog()
                    ),
                    MDFlatButton(
                        text="CONFIRM",
                        on_release=lambda x: (self.dismiss_dialog(), on_confirm())
                    )
                ]
            )
            self.dialog.open()
        Clock.schedule_once(create_dialog, 0)
    
    # =========================================================================
    # Navigation Methods
    # =========================================================================
    
    def go_to_screen(self, screen_name: str):
        """Navigate to a specific screen."""
        self.screen_manager.current = screen_name
    
    # =========================================================================
    # Batch Control Methods
    # =========================================================================
    
    def reset_batch(self):
        """Reset current batch count."""
        def confirm_reset():
            self.show_loading("Resetting batch...")
            
            def callback(success, data):
                def ui_update(dt):
                    self.dismiss_dialog()
                    if success:
                        self.show_success("Batch reset successfully!")
                        # Refresh dashboard
                        dashboard = self.screen_manager.get_screen('dashboard')
                        dashboard.refresh_data()
                    else:
                        self.show_error("Failed to reset batch")
                
                Clock.schedule_once(ui_update, 0)
            
            self.api_client.reset_batch(callback=callback)
        
        self.show_confirmation(
            "Reset Batch",
            "Are you sure you want to reset the current batch count to 0?",
            confirm_reset
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    CaneToadDetectorApp().run()
