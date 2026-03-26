# =============================================================================
# LOGIN SCREEN
# =============================================================================

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.app import App

Builder.load_string("""
<LoginScreen>:
    name: 'login'
    
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(40)
        spacing: dp(20)
        md_bg_color: app.theme_cls.bg_dark
        
        # Logo/Title Section
        MDBoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.3
            spacing: dp(10)
            
            MDIcon:
                icon: 'frog'
                halign: 'center'
                font_size: '80sp'
                theme_text_color: 'Custom'
                text_color: app.theme_cls.primary_color
            
            MDLabel:
                text: 'Cane Toad Detector'
                halign: 'center'
                font_style: 'H4'
                theme_text_color: 'Primary'
            
            MDLabel:
                text: 'Agricultural Pest Management System'
                halign: 'center'
                font_style: 'Caption'
                theme_text_color: 'Hint'
        
        # Form Section
        MDBoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.5
            spacing: dp(15)
            
            MDTextField:
                id: username_field
                hint_text: 'Username'
                icon_left: 'account'
                mode: 'rectangle'
                size_hint_x: 1
                
            MDTextField:
                id: password_field
                hint_text: 'Password'
                icon_left: 'lock'
                password: True
                mode: 'rectangle'
                size_hint_x: 1
            
            MDRaisedButton:
                text: 'LOGIN'
                size_hint_x: 1
                md_bg_color: app.theme_cls.primary_color
                on_release: root.do_login()
            
            MDFlatButton:
                text: "Don't have an account? Register"
                size_hint_x: 1
                on_release: root.go_to_register()
            
            MDFlatButton:
                text: "⚙️  Connection Settings"
                size_hint_x: 1
                on_release: root.go_to_settings()
        
        # Spacer
        MDBoxLayout:
            size_hint_y: 0.2
""")


class LoginScreen(Screen):
    """Login screen with username/password authentication."""
    
    def do_login(self):
        """Handle login button press."""
        username = self.ids.username_field.text.strip()
        password = self.ids.password_field.text.strip()
        
        # Call app's login method
        App.get_running_app().login(username, password)
    
    def go_to_register(self):
        """Navigate to registration screen."""
        self.manager.current = 'register'
    
    def go_to_settings(self):
        """Navigate to settings screen."""
        self.manager.current = 'settings'
    
    def on_enter(self):
        """Called when screen is entered."""
        # Clear password field for security
        self.ids.password_field.text = ''
