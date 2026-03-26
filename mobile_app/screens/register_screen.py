# =============================================================================
# REGISTRATION SCREEN
# =============================================================================

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.app import App

Builder.load_string("""
<RegisterScreen>:
    name: 'register'
    
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(40)
        spacing: dp(20)
        md_bg_color: app.theme_cls.bg_dark
        
        # Header
        MDBoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.2
            spacing: dp(10)
            
            MDIconButton:
                icon: 'arrow-left'
                pos_hint: {'left': 1}
                on_release: root.go_back()
            
            MDLabel:
                text: 'Create Account'
                halign: 'center'
                font_style: 'H5'
                theme_text_color: 'Primary'
            
            MDLabel:
                text: 'Register for cane toad monitoring'
                halign: 'center'
                font_style: 'Caption'
                theme_text_color: 'Hint'
        
        # Form Section
        MDBoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.6
            spacing: dp(15)
            
            MDTextField:
                id: username_field
                hint_text: 'Username (min 3 characters)'
                icon_left: 'account'
                mode: 'rectangle'
                size_hint_x: 1
                
            MDTextField:
                id: password_field
                hint_text: 'Password (min 6 characters)'
                icon_left: 'lock'
                password: True
                mode: 'rectangle'
                size_hint_x: 1
            
            MDTextField:
                id: confirm_password_field
                hint_text: 'Confirm Password'
                icon_left: 'lock-check'
                password: True
                mode: 'rectangle'
                size_hint_x: 1
            
            MDRaisedButton:
                text: 'CREATE ACCOUNT'
                size_hint_x: 1
                md_bg_color: app.theme_cls.primary_color
                on_release: root.do_register()
            
            MDFlatButton:
                text: 'Already have an account? Login'
                size_hint_x: 1
                on_release: root.go_back()
        
        # Spacer
        MDBoxLayout:
            size_hint_y: 0.2
""")


class RegisterScreen(Screen):
    """Registration screen for new users."""
    
    def do_register(self):
        """Handle registration button press."""
        username = self.ids.username_field.text.strip()
        password = self.ids.password_field.text.strip()
        confirm = self.ids.confirm_password_field.text.strip()
        
        # Call app's register method
        App.get_running_app().register(username, password, confirm)
    
    def go_back(self):
        """Navigate back to login screen."""
        self.manager.current = 'login'
    
    def on_enter(self):
        """Called when screen is entered."""
        # Clear all fields
        self.ids.username_field.text = ''
        self.ids.password_field.text = ''
        self.ids.confirm_password_field.text = ''
