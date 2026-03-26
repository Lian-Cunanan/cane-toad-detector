# =============================================================================
# DASHBOARD SCREEN
# =============================================================================

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.app import App

Builder.load_string("""
<DashboardScreen>:
    name: 'dashboard'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_dark
        
        # Top Bar
        MDTopAppBar:
            title: 'Cane Toad Detector'
            left_action_items: [['menu', lambda x: None]]
            right_action_items: [['camera', lambda x: root.go_to_camera()]]
        
        # Scrollable Content
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: dp(20)
                spacing: dp(15)
                
                # Welcome Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(15)
                    size_hint_y: None
                    height: dp(80)
                    
                    MDLabel:
                        text: 'Cane Toad Detector'
                        font_style: 'H6'
                        theme_text_color: 'Primary'
                    
                    MDLabel:
                        text: root.current_phase_text
                        font_style: 'Body1'
                        theme_text_color: 'Hint'
                
                # Progress Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    size_hint_y: None
                    height: dp(200)
                    
                    MDLabel:
                        text: 'Capture Progress'
                        font_style: 'H6'
                        theme_text_color: 'Primary'
                        size_hint_y: 0.2
                    
                    MDBoxLayout:
                        orientation: 'horizontal'
                        size_hint_y: 0.3
                        spacing: dp(10)
                        
                        MDLabel:
                            text: str(root.current_count)
                            font_style: 'H3'
                            halign: 'center'
                            theme_text_color: 'Custom'
                            text_color: app.theme_cls.primary_color
                        
                        MDLabel:
                            text: '/'
                            font_style: 'H4'
                            halign: 'center'
                        
                        MDLabel:
                            text: str(root.target_count)
                            font_style: 'H3'
                            halign: 'center'
                    
                    MDProgressBar:
                        value: root.progress_percent
                        size_hint_y: 0.2
                    
                    MDLabel:
                        text: f'{int(root.progress_percent)}% Complete'
                        halign: 'center'
                        font_style: 'Caption'
                        size_hint_y: 0.3
                
                # Operational Phase Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    size_hint_y: None
                    height: dp(180)
                    
                    MDLabel:
                        text: 'Operational Phase'
                        font_style: 'H6'
                        theme_text_color: 'Primary'
                        size_hint_y: 0.25
                    
                    # Phase Indicators
                    MDGridLayout:
                        cols: 2
                        spacing: dp(10)
                        size_hint_y: 0.75
                        
                        PhaseIndicator:
                            phase_name: 'Capturing'
                            is_active: root.current_phase == 'Capturing'
                            icon: 'camera-iris'
                        
                        PhaseIndicator:
                            phase_name: 'Euthanizing'
                            is_active: root.current_phase == 'Euthanizing'
                            icon: 'gas-cylinder'
                        
                        PhaseIndicator:
                            phase_name: 'Disposing'
                            is_active: root.current_phase == 'Disposing'
                            icon: 'delete'
                        
                        PhaseIndicator:
                            phase_name: 'Heat Sealing'
                            is_active: root.current_phase == 'Heat Sealing'
                            icon: 'package-variant-closed'
                
                # System Status Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    size_hint_y: None
                    height: dp(200)
                    
                    MDLabel:
                        text: 'System Status'
                        font_style: 'H6'
                        theme_text_color: 'Primary'
                        size_hint_y: 0.2
                    
                    MDBoxLayout:
                        orientation: 'vertical'
                        spacing: dp(10)
                        size_hint_y: 0.8
                        
                        StatusRow:
                            icon: 'battery'
                            label: 'Battery'
                            value: f'{root.battery_level}%'
                            is_good: root.battery_level > 20
                        
                        StatusRow:
                            icon: 'wifi'
                            label: 'WiFi'
                            value: 'Connected' if root.wifi_connected else 'Disconnected'
                            is_good: root.wifi_connected
                        
                        StatusRow:
                            icon: 'camera'
                            label: 'Camera'
                            value: 'Active' if root.camera_active else 'Inactive'
                            is_good: root.camera_active
                        
                        StatusRow:
                            icon: 'clock-outline'
                            label: 'Last Detection'
                            value: root.last_detection_text
                            is_good: True
                
                # Control Buttons
                MDBoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(120)
                    spacing: dp(10)
                    
                    MDRaisedButton:
                        text: 'NEXT BATCH / RESET'
                        size_hint_x: 1
                        md_bg_color: app.theme_cls.accent_color
                        on_release: root.reset_batch()
                    
                    MDRaisedButton:
                        text: 'VIEW CAMERA FEEDS'
                        size_hint_x: 1
                        md_bg_color: app.theme_cls.primary_color
                        on_release: root.go_to_camera()

<PhaseIndicator@MDCard>:
    phase_name: ''
    is_active: False
    icon: 'help'
    
    orientation: 'vertical'
    padding: dp(10)
    md_bg_color: app.theme_cls.primary_color if self.is_active else app.theme_cls.bg_light
    
    MDIcon:
        icon: root.icon
        halign: 'center'
        size_hint_y: 0.6
        theme_text_color: 'Custom'
        text_color: [1, 1, 1, 1] if root.is_active else app.theme_cls.disabled_hint_text_color
    
    MDLabel:
        text: root.phase_name
        halign: 'center'
        font_style: 'Caption'
        size_hint_y: 0.4
        theme_text_color: 'Custom'
        text_color: [1, 1, 1, 1] if root.is_active else app.theme_cls.disabled_hint_text_color

<StatusRow@MDBoxLayout>:
    icon: ''
    label: ''
    value: ''
    is_good: True
    
    orientation: 'horizontal'
    spacing: dp(10)
    size_hint_y: None
    height: dp(30)
    
    MDIcon:
        icon: root.icon
        size_hint_x: 0.15
        theme_text_color: 'Custom'
        text_color: [0, 1, 0, 1] if root.is_good else [1, 0, 0, 1]
    
    MDLabel:
        text: root.label
        size_hint_x: 0.45
        font_style: 'Body2'
    
    MDLabel:
        text: root.value
        size_hint_x: 0.4
        halign: 'right'
        font_style: 'Body2'
        theme_text_color: 'Custom'
        text_color: [0, 1, 0, 1] if root.is_good else [1, 0, 0, 1]
""")


class DashboardScreen(Screen):
    """Main dashboard screen with real-time monitoring."""
    
    # Progress properties
    current_count = NumericProperty(0)
    target_count = NumericProperty(10)
    progress_percent = NumericProperty(0)
    
    # Phase properties
    current_phase = StringProperty('Idle')
    current_phase_text = StringProperty('System ready')
    
    # Status properties
    battery_level = NumericProperty(100)
    wifi_connected = BooleanProperty(True)
    camera_active = BooleanProperty(False)
    last_detection_text = StringProperty('No detections yet')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refresh_event = None
    
    def on_enter(self):
        """Called when screen is entered."""
        # Set up WebSocket callbacks
        app = App.get_running_app()
        if app.api_client:
            app.api_client.set_detection_callback(self.on_detection_alert)
            app.api_client.set_status_update_callback(self.on_status_update)
            app.api_client.set_phase_update_callback(self.on_phase_update)
        
        # Start periodic refresh
        self.refresh_data()
        self.refresh_event = Clock.schedule_interval(lambda dt: self.refresh_data(), 5)
    
    def on_leave(self):
        """Called when leaving screen."""
        if self.refresh_event:
            self.refresh_event.cancel()
    
    def refresh_data(self):
        """Refresh all dashboard data from backend."""
        app = App.get_running_app()
        
        if not app.api_client:
            return
        
        # Get current detections
        def detection_callback(success, data):
            if success:
                self.current_count = data.get('current_count', 0)
                self.target_count = data.get('target_count', 10)
                self.progress_percent = (self.current_count / self.target_count * 100) if self.target_count > 0 else 0
                self.current_phase = data.get('phase', 'Idle')
                self.current_phase_text = f'Phase: {self.current_phase}'
        
        app.api_client.get_current_detections(callback=detection_callback)
        
        # Get system status
        def status_callback(success, data):
            if success:
                self.battery_level = data.get('battery_level', 100)
                self.wifi_connected = data.get('wifi_connected', True)
                self.camera_active = data.get('camera_active', False)
                
                last_det = data.get('last_detection_time')
                if last_det:
                    # Format timestamp
                    self.last_detection_text = last_det.split('T')[1][:8]  # Show time only
                else:
                    self.last_detection_text = 'No detections yet'
        
        app.api_client.get_system_status(callback=status_callback)
    
    # =========================================================================
    # WebSocket Event Handlers
    # =========================================================================
    
    def on_detection_alert(self, data):
        """Handle real-time detection alert."""
        def ui_update(dt):
            self.current_count = data.get('count', self.current_count)
            self.target_count = data.get('target', self.target_count)
            self.progress_percent = (self.current_count / self.target_count * 100) if self.target_count > 0 else 0
            
            # Show notification
            app = App.get_running_app()
            app.show_info(f"Toad detected! Count: {self.current_count}/{self.target_count}")
        
        Clock.schedule_once(ui_update, 0)
    
    def on_status_update(self, data):
        """Handle real-time status update."""
        def ui_update(dt):
            self.battery_level = data.get('battery_level', self.battery_level)
            self.wifi_connected = data.get('wifi_connected', self.wifi_connected)
            self.camera_active = data.get('camera_active', self.camera_active)
        
        Clock.schedule_once(ui_update, 0)
    
    def on_phase_update(self, data):
        """Handle real-time phase update."""
        def ui_update(dt):
            self.current_phase = data.get('phase_name', 'Idle')
            self.current_phase_text = f'Phase: {self.current_phase}'
        
        Clock.schedule_once(ui_update, 0)
    
    # =========================================================================
    # Navigation
    # =========================================================================
    
    def go_to_camera(self):
        """Navigate to camera screen."""
        self.manager.current = 'camera'
    
    def reset_batch(self):
        """Reset current batch."""
        App.get_running_app().reset_batch()
