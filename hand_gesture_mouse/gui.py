import threading
import tkinter as tk
from tkinter import ttk
import pystray
from PIL import Image, ImageDraw

from . import settings

class SettingsGUI:
    def __init__(self):
        self.root = None

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Hand Gesture Mouse Settings")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Sensitivity Settings", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))

        self._create_slider(main_frame, "Scroll Sensitivity", "SCROLL_SENSITIVITY", 1, 20)
        self._create_slider(main_frame, "Cursor Smoothing", "CURSOR_SMOOTH_FRAMES", 1, 10, value_type=int)
        self._create_slider(main_frame, "Right Click Hold (sec)", "RIGHT_CLICK_HOLD_SEC", 0.1, 2.0, resolution=0.1)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        ttk.Label(main_frame, text="General Settings", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))
        
        self.fps_var = tk.BooleanVar(value=settings.DISPLAY_FPS)
        fps_cb = ttk.Checkbutton(main_frame, text="Display HUD overlay on camera", variable=self.fps_var, command=self._on_fps_toggle)
        fps_cb.pack(anchor=tk.W, pady=5)

        ttk.Button(main_frame, text="Close", command=self.root.destroy).pack(side=tk.BOTTOM, pady=20)

        # Handle window close gracefully so we can open it again later
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _create_slider(self, parent, label_text, setting_key, from_, to, resolution=1, value_type=float):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
        
        val_var = tk.DoubleVar(value=getattr(settings, setting_key))
        val_label = ttk.Label(frame, text=f"{val_var.get():.2f}")
        val_label.pack(side=tk.RIGHT)
        
        def on_change(event=None):
            val = val_var.get()
            if value_type == int:
                val = int(val)
            val_label.config(text=f"{val:.2f}" if value_type == float else f"{val}")
            setattr(settings, setting_key, val)
        
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=val_var, command=on_change)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

    def _on_fps_toggle(self):
        settings.DISPLAY_FPS = self.fps_var.get()

    def _on_close(self):
        self.root.destroy()
        self.root = None

    def show(self):
        if self.root is None or not self.root.winfo_exists():
            self._build_ui()
            self.root.mainloop()
        else:
            self.root.lift()

def create_tray_icon():
    # Create simple generic icon for the system tray
    width = 64
    height = 64
    image = Image.new('RGB', (width, height), color=(30, 30, 30))
    dc = ImageDraw.Draw(image)
    dc.ellipse((10, 10, 54, 54), fill=(0, 200, 120))
    dc.polygon([(32, 20), (45, 45), (19, 45)], fill=(255, 255, 255))
    
    gui_manager = SettingsGUI()
    
    def on_settings(icon, item):
        # Must run UI on a thread so Windows event loop doesn't block system tray
        threading.Thread(target=gui_manager.show, daemon=True).start()
        
    def on_quit(icon, item):
        from .state import try_put_ctrl
        try_put_ctrl(("quit",))
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Settings...", on_settings, default=True),
        pystray.MenuItem("Quit", on_quit)
    )
    
    icon = pystray.Icon("HandGestureMouse", image, "Hand Gesture Mouse", menu)
    return icon

def start_tray_detached():
    icon = create_tray_icon()
    threading.Thread(target=icon.run, daemon=True).start()
    return icon
