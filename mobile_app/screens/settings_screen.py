# =============================================================================
# SETTINGS SCREEN
# =============================================================================

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.app import App
from kivy.storage.jsonstore import JsonStore
from kivy.properties import StringProperty
from pathlib import Path

Builder.load_string("""
<SettingsScreen>:
    name: 'settings'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_dark
        
        # Top Bar
        MDTopAppBar:
            title: 'Settings'
            left_action_items: [['arrow-left', lambda x: root.go_back()]]
        
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: dp(20)
                spacing: dp(20)
                
                # Connection Settings Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(15)
                    size_hint_y: None
                    height: dp(280)
                    
                    MDLabel:
                        text: 'Connection Settings'
                        font_style: 'H6'
                        size_hint_y: None
                        height: dp(40)
                        theme_text_color: 'Primary'
                    
                    MDLabel:
                        text: 'Backend Server URL'
                        font_style: 'Body2'
                        size_hint_y: None
                        height: dp(30)
                        theme_text_color: 'Hint'
                    
                    MDTextField:
                        id: backend_url_field
                        text: root.backend_url
                        hint_text: 'http://192.168.4.1:5000'
                        mode: 'rectangle'
                        helper_text: 'Enter Raspberry Pi hotspot IP'
                        helper_text_mode: 'on_focus'
                        size_hint_x: 1
                    
                    MDBoxLayout:
                        size_hint_y: None
                        height: dp(50)
                        spacing: dp(10)
                        
                        MDRaisedButton:
                            text: 'TEST CONNECTION'
                            size_hint_x: 0.5
                            on_release: root.test_connection()
                        
                        MDRaisedButton:
                            text: 'SAVE'
                            size_hint_x: 0.5
                            md_bg_color: app.theme_cls.primary_color
                            on_release: root.save_settings()
                    
                    MDLabel:
                        id: connection_status
                        text: root.connection_status
                        font_style: 'Caption'
                        size_hint_y: None
                        height: dp(30)
                        theme_text_color: 'Hint'
                
                # Quick Presets Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(15)
                    size_hint_y: None
                    height: dp(220)
                    
                    MDLabel:
                        text: 'Quick Presets'
                        font_style: 'H6'
                        size_hint_y: None
                        height: dp(40)
                        theme_text_color: 'Primary'
                    
                    MDRaisedButton:
                        text: 'Raspberry Pi Hotspot (192.168.4.1:5000)'
                        size_hint_x: 1
                        on_release: root.set_preset('http://192.168.4.1:5000')
                    
                    MDRaisedButton:
                        text: 'Local Network (192.168.1.100:5000)'
                        size_hint_x: 1
                        on_release: root.set_preset('http://192.168.1.100:5000')
                    
                    MDRaisedButton:
                        text: 'Localhost (http://localhost:5000)'
                        size_hint_x: 1
                        on_release: root.set_preset('http://localhost:5000')
                
                # About Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(15)
                    size_hint_y: None
                    height: dp(180)
                    
                    MDLabel:
                        text: 'About'
                        font_style: 'H6'
                        size_hint_y: None
                        height: dp(40)
                        theme_text_color: 'Primary'
                    
                    MDLabel:
                        text: 'Cane Toad Detector'
                        font_style: 'Body1'
                        theme_text_color: 'Primary'
                    
                    MDLabel:
                        text: 'Version 1.0.0'
                        font_style: 'Caption'
                        theme_text_color: 'Hint'
                    
                    MDLabel:
                        text: 'Agricultural Pest Management System'
                        font_style: 'Caption'
                        theme_text_color: 'Hint'
""")


class SettingsScreen(Screen):
    """Settings screen for configuring backend connection."""
    backend_url = StringProperty('')
    connection_status = StringProperty('')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = None
        
    def on_enter(self):
        """Called when screen is entered."""
        # Initialize storage
        app = App.get_running_app()
        storage_path = str(Path.home() / '.cane_toad_detector')
        Path(storage_path).mkdir(exist_ok=True)
        self.store = JsonStore(str(Path(storage_path) / 'settings.json'))
        
        # Load current backend URL
        self.backend_url = app.backend_url
        self.ids.backend_url_field.text = self.backend_url
        self.connection_status = ''
    
    def set_preset(self, url: str):
        """Set a preset URL."""
        self.ids.backend_url_field.text = url
        self.backend_url = url
    
    def test_connection(self):
        """Test connection to backend."""
        import requests
        from kivy.clock import Clock
        
        url = self.ids.backend_url_field.text.strip()
        if not url:
            self.connection_status = '❌ Please enter a URL'
            return
        
        self.connection_status = '⏳ Testing connection...'
        
        def do_test(dt):
            try:
                response = requests.get(f"{url}/api/auth/register", timeout=3)
                if response.status_code in [200, 400, 405]:  # Expected responses
                    self.connection_status = f'✅ Connected to {url}'
                else:
                    self.connection_status = f'⚠️  Server responded with {response.status_code}'
            except requests.exceptions.Timeout:
                self.connection_status = '❌ Connection timeout - Check IP address'
            except requests.exceptions.ConnectionError:
                self.connection_status = '❌ Connection failed - Is Pi powered on?'
            except Exception as e:
                self.connection_status = f'❌ Error: {str(e)[:50]}'
        
        Clock.schedule_once(do_test, 0.1)
    
    def save_settings(self):
        """Save settings and update app."""
        url = self.ids.backend_url_field.text.strip()
        
        if not url:
            app = App.get_running_app()
            app.show_error("Please enter a backend URL")
            return
        
        # Validate URL format
        if not url.startswith('http://') and not url.startswith('https://'):
            app = App.get_running_app()
            app.show_error("URL must start with http:// or https://")
            return
        
        # Save to storage
        if self.store:
            self.store.put('backend', url=url)
        
        # Update app
        app = App.get_running_app()
        app.backend_url = url
        
        # Update API client
        if app.api_client:
            app.api_client.base_url = url.rstrip('/')
            app.api_client.api_url = f"{app.api_client.base_url}/api"
        
        app.show_success(f"Settings saved!\nBackend: {url}")
        self.go_back()
    
    def go_back(self):
        """Navigate back to previous screen."""
        self.manager.current = 'login'
