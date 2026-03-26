# =============================================================================
# MJPEG Viewer Widget for Kivy
# =============================================================================
#
# Custom widget to display MJPEG video streams in Kivy applications.
# AsyncImage doesn't support MJPEG, so we need this custom implementation.
# =============================================================================

import threading
import requests
from io import BytesIO
from PIL import Image as PILImage

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class MJPEGViewer(Image):
    """
    Custom Kivy widget to display MJPEG streams.
    
    Usage:
        viewer = MJPEGViewer()
        viewer.start_stream("http://localhost:5000/api/camera/cage/stream?token=...")
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream_url = None
        self.stream_thread = None
        self.is_streaming = False
        self.allow_stretch = True
        self.keep_ratio = True
        
    def start_stream(self, url: str):
        """Start streaming from the given MJPEG URL."""
        # Stop existing stream if any
        self.stop_stream()
        
        self.stream_url = url
        self.is_streaming = True
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
    
    def stop_stream(self):
        """Stop the current stream."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None
    
    def _stream_loop(self):
        """Background thread that fetches MJPEG frames."""
        try:
            response = requests.get(self.stream_url, stream=True, timeout=10)
            
            if response.status_code != 200:
                print(f"[MJPEG] Stream error: {response.status_code}")
                return
            
            # Parse multipart stream
            bytes_data = b''
            for chunk in response.iter_content(chunk_size=1024):
                if not self.is_streaming:
                    break
                
                bytes_data += chunk
                
                # Look for JPEG markers
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    # Extract JPEG frame
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode and display frame
                    try:
                        pil_image = PILImage.open(BytesIO(jpg))
                        Clock.schedule_once(lambda dt, img=pil_image: self._update_texture(img), 0)
                    except Exception as e:
                        print(f"[MJPEG] Frame decode error: {e}")
                        
        except Exception as e:
            print(f"[MJPEG] Stream error: {e}")
        finally:
            self.is_streaming = False
    
    def _update_texture(self, pil_image):
        """Update the texture with a new frame (must be called on main thread)."""
        try:
            # Convert PIL image to Kivy texture
            image_data = pil_image.convert('RGB').tobytes()
            texture = Texture.create(size=pil_image.size, colorfmt='rgb')
            texture.blit_buffer(image_data, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            
            self.texture = texture
            self.canvas.ask_update()
        except Exception as e:
            print(f"[MJPEG] Texture update error: {e}")
    
    def on_parent(self, widget, parent):
        """Called when widget is added/removed from parent."""
        if parent is None:
            # Widget removed - stop streaming
            self.stop_stream()
