# =============================================================================
# CAMERA SCREEN
# =============================================================================

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.app import App

# Import custom MJPEG viewer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'widgets'))
from mjpeg_viewer import MJPEGViewer

Builder.load_string("""
<CameraScreen>:
    name: 'camera'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_dark
        
        # Top Bar
        MDTopAppBar:
            title: 'Camera Feeds'
            left_action_items: [['arrow-left', lambda x: root.go_back()]]
            right_action_items: [['refresh', lambda x: root.refresh_feeds()]]
        
        # Camera Feeds
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: dp(10)
                spacing: dp(15)
                
                # Cage Camera Card
                MDCard:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(320)
                    padding: dp(10)
                    
                    MDLabel:
                        text: 'Cage View'
                        font_style: 'H6'
                        size_hint_y: 0.1
                        theme_text_color: 'Primary'
                    
                    MDBoxLayout:
                        id: cage_camera_container
                        size_hint_y: 0.8
                    
                    MDLabel:
                        id: cage_status
                        text: root.cage_status_text
                        font_style: 'Caption'
                        size_hint_y: 0.1
                        halign: 'center'
                        theme_text_color: 'Hint'
                
                # Trap Camera Card
                MDCard:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(320)
                    padding: dp(10)
                    
                    MDLabel:
                        text: 'Trap View'
                        font_style: 'H6'
                        size_hint_y: 0.1
                        theme_text_color: 'Primary'
                    
                    MDBoxLayout:
                        id: trap_camera_container
                        size_hint_y: 0.8
                    
                    MDLabel:
                        id: trap_status
                        text: root.trap_status_text
                        font_style: 'Caption'
                        size_hint_y: 0.1
                        halign: 'center'
                        theme_text_color: 'Hint'
                
                # Controls
                MDBoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: dp(50)
                    spacing: dp(10)
                    
                    MDRaisedButton:
                        text: 'BACK TO DASHBOARD'
                        size_hint_x: 1
                        md_bg_color: app.theme_cls.primary_color
                        on_release: root.go_back()
""")


class CameraScreen(Screen):
    """Screen displaying dual camera feeds."""
    
    # Kivy properties for binding to UI
    cage_status_text = StringProperty('Connecting...')
    trap_status_text = StringProperty('Connecting...')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refresh_event = None
        self.cage_viewer = None
        self.trap_viewer = None
    
    def on_enter(self):
        """Called when screen is entered."""
        # Create MJPEG viewers
        self.cage_viewer = MJPEGViewer()
        self.trap_viewer = MJPEGViewer()
        
        # Add viewers to containers
        self.ids.cage_camera_container.clear_widgets()
        self.ids.cage_camera_container.add_widget(self.cage_viewer)
        
        self.ids.trap_camera_container.clear_widgets()
        self.ids.trap_camera_container.add_widget(self.trap_viewer)
        
        # Start streams
        self.setup_camera_urls()
        self.check_camera_status()
        
        # Periodically re-check camera status text (stream itself is handled by MJPEGViewer)
        self.refresh_event = Clock.schedule_interval(lambda dt: self.refresh_feeds(), 10)
    
    def on_leave(self):
        """Called when leaving screen."""
        if self.refresh_event:
            self.refresh_event.cancel()
        
        # Stop camera streams
        if self.cage_viewer:
            self.cage_viewer.stop_stream()
        if self.trap_viewer:
            self.trap_viewer.stop_stream()
    
    def setup_camera_urls(self):
        """Set up camera stream URLs and start streaming."""
        app = App.get_running_app()
        
        if not app.api_client or not app.access_token:
            self.cage_status_text = 'Not authenticated'
            self.trap_status_text = 'Not authenticated'
            return
        
        # Build URLs with authentication token as query parameter
        base_url = app.backend_url
        token = app.access_token
        
        cage_url = f"{base_url}/api/camera/cage/stream?token={token}"
        trap_url = f"{base_url}/api/camera/trap/stream?token={token}"
        
        # Start MJPEG streams
        if self.cage_viewer:
            self.cage_viewer.start_stream(cage_url)
            self.cage_status_text = 'Streaming...'
        
        if self.trap_viewer:
            self.trap_viewer.start_stream(trap_url)
            self.trap_status_text = 'Streaming...'
    
    def check_camera_status(self):
        """Check if cameras are available."""
        app = App.get_running_app()
        
        if not app.api_client:
            return
        
        def status_callback(success, data):
            if success:
                cage_active = data.get('cage_camera', False)
                trap_active = data.get('trap_camera', False)
                
                self.cage_status_text = 'Live' if cage_active else 'Camera unavailable'
                self.trap_status_text = 'Live' if trap_active else 'Camera unavailable'
            else:
                self.cage_status_text = 'Connection error'
                self.trap_status_text = 'Connection error'
        
        app.api_client.get_camera_status(callback=status_callback)
    
    def refresh_feeds(self):
        """Re-check camera status labels. The MJPEGViewer handles streaming itself."""
        self.check_camera_status()
    
    def go_back(self):
        """Return to dashboard."""
        self.manager.current = 'dashboard'
