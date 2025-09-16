"""
Spektrometr Application - Main File
Refactored with proper threading and structure
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import json
import csv
import glob

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
import serial.tools.list_ports

# Local imports
from pixelinkWrapper import *
from async_tkinter_loop import async_mainloop

# Load configuration
try:
    with open('options.json', 'r') as f:
        options = json.load(f)
except FileNotFoundError:
    options = {
        'step_x': 10, 'step_y': 10, 'offset': 5,
        'width': 100, 'height': 100, 'await': 0.01,
        'sequence_sleep': 0.1,  # Sleep time during sequence measurements
        'xmin': '0', 'xmax': '2048',
        'port_x': 'COM5', 'port_y': 'COM9',
        'camera_index': 0,  # Try camera 0 by default
        'cal_px_per_step_x': 0.0, 'cal_px_per_step_y': 0.0,
        'cal_sign_x': 1, 'cal_sign_y': 1,
        'cal_step_x': 1, 'cal_step_y': 1
    }

# Color constants
LGRAY = '#232323'
DGRAY = '#161616'
RGRAY = '#2c2c2c'
MGRAY = '#1D1c1c'


class StreamToFunction:
    """Redirect stdout to a function"""
    def __init__(self, func):
        self.func = func

    def write(self, message):
        if message.strip():
            self.func(message)

    def flush(self):
        pass


class CameraManager:
    """Manages camera operations in separate thread"""
    
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.detector = None
        self.running = False
        self.thread = None
        self.frame = None
        self.direction = "No movement"
        
    def start(self):
        """Start camera thread"""
        if not self.running:
            self.running = True
            self.detector = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            # Check if camera opened successfully
            if not self.detector.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                self.detector = None
                self.running = False
                return False
                
            self.thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        if self.detector:
            self.detector.release()
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _camera_loop(self):
        """Main camera loop running in thread"""
        prev_frame = None
        
        while self.running:
            try:
                # Check if detector is valid before reading
                if not self.detector or not self.detector.isOpened():
                    time.sleep(0.1)
                    continue
                    
                ret, frame = self.detector.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                
                # Motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                direction = "No movement"
                
                if prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mean_flow = np.mean(flow, axis=(0, 1))
                    magnitude = np.linalg.norm(mean_flow)
                    
                    if magnitude > 0.1:
                        if abs(mean_flow[0]) > abs(mean_flow[1]):
                            direction = 'Right' if mean_flow[0] > 0 else 'Left'
                        else:
                            direction = 'Down' if mean_flow[1] > 0 else 'Up'
                
                prev_frame = gray
                
                # Add crosshair
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150), 1)
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150), 1)
                
                # Store frame and direction directly
                self.direction = direction
                self.frame = frame
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)
    
    def get_current_frame(self):
        """Get the current frame from camera"""
        return self.frame
    
    def get_current_direction(self):
        """Get the current movement direction"""
        return self.direction


class SpectrometerManager:
    """Simplified Pixelink camera manager based on samples/getNextNumPyFrame.py"""
    
    def __init__(self):
        self.hCamera = None
        self.running = False
        self.thread = None
        
        # Create buffer exactly like sample
        MAX_WIDTH = 5000   # in pixels
        MAX_HEIGHT = 5000  # in pixels
        MAX_BYTES_PER_PIXEL = 3
        self.frame_buffer = np.zeros([MAX_HEIGHT, MAX_WIDTH*MAX_BYTES_PER_PIXEL], dtype=np.uint8)
        
        print(f"✅ SpectrometerManager buffer created: {self.frame_buffer.shape}")
        
    def initialize(self):
        """Initialize camera exactly like sample"""
        try:
            print("🔄 Initializing PixeLink camera (sample style)...")
            
            # Initialize any camera - exactly like samples
            ret = PxLApi.initialize(0)
            if not PxLApi.apiSuccess(ret[0]):
                print(f"❌ Error: Unable to initialize a camera! rc = {ret[0]}")
                print("🔄 Loading test image instead...")
                self._load_test_image()
                return False

            self.hCamera = ret[1]
            print(f"✅ Camera initialized successfully. Handle: {self.hCamera}")
            return True
                
        except Exception as e:
            print(f"❌ Pixelink initialization error: {e}")
            print("🔄 Loading test image instead...")
            self._load_test_image()
            return False
    
    def _load_test_image(self):
        """Load test image when camera is not available"""
        try:
            import cv2
            test_image_path = "2.bmp"
            if os.path.exists(test_image_path):
                # Load test image using OpenCV
                test_img = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
                if test_img is not None:
                    # Convert to the same format as camera buffer
                    if len(test_img.shape) == 3:
                        # Convert BGR to grayscale or flatten
                        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize/reshape to fit frame_buffer dimensions if needed
                    h, w = test_img.shape
                    if h * w <= self.frame_buffer.size:
                        # Flatten and copy to frame buffer
                        flat_img = test_img.flatten()
                        self.frame_buffer.flat[:len(flat_img)] = flat_img
                        print(f"✅ Test image loaded: {test_image_path} ({h}x{w})")
                    else:
                        print(f"⚠️ Test image too large: {h}x{w}")
                else:
                    print(f"❌ Could not load test image: {test_image_path}")
            else:
                print(f"❌ Test image not found: {test_image_path}")
        except Exception as e:
            print(f"❌ Error loading test image: {e}")
    
    def start(self):
        """Start streaming exactly like sample"""
        if self.hCamera and not self.running:
            print("🔄 Starting PixeLink streaming...")
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("✅ PixeLink thread started")
    
    def stop(self):
        """Stop streaming exactly like sample"""
        print("🔄 Stopping PixeLink streaming...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if self.hCamera:
            try:
                PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
                PxLApi.uninitialize(self.hCamera)
                print("✅ PixeLink camera stopped and uninitialized")
            except Exception as e:
                print(f"❌ Error stopping PixeLink: {e}")
                
        self.hCamera = None

    def get_next_frame(self, maxTries=5):
        """Robust wrapper around getNextFrame exactly like sample"""
        ret = (PxLApi.ReturnCode.ApiUnknownError,)
        
        for _ in range(maxTries):
            ret = PxLApi.getNextNumPyFrame(self.hCamera, self.frame_buffer)
            if PxLApi.apiSuccess(ret[0]):
                return ret
            else:
                # If the streaming is turned off, or worse yet -- is gone?
                if PxLApi.ReturnCode.ApiStreamStopped == ret[0] or \
                   PxLApi.ReturnCode.ApiNoCameraAvailableError == ret[0]:
                    return ret
                else:
                    print(f"    Hmmm... getNextFrame returned {ret[0]}")
        
        # Ran out of tries
        return ret

    def _capture_loop(self):
        """Main capture loop - exactly like sample getNextNumPyFrame.py"""
        if not self.hCamera or not self.frame_buffer.size:
            return
            
        # Start the stream exactly like sample
        ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)
        if not PxLApi.apiSuccess(ret[0]):
            print(f"setStreamState with StreamState.START failed, rc = {ret[0]}")
            return
            
        print("✅ PixeLink streaming started successfully (sample style)")

        while self.running:
            try:
                # Use robust wrapper exactly like sample
                ret = self.get_next_frame(1)
                
                if PxLApi.apiSuccess(ret[0]):
                    self.frame_buffer = ret[1]

                time.sleep(0.5)  # 500ms like sample
                
            except Exception as e:
                print(f"❌ PixeLink capture error: {e}")
                # Load test image on error
                self._load_test_image()
                time.sleep(0.1)
        
        # Stop streaming when loop ends
        try:
            ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
            print("✅ Streaming stopped")
        except Exception as e:
            print(f"❌ Stop streaming error: {e}")


class MotorController:
    """Controls stepper motors"""
    
    def __init__(self, port_x='COM5', port_y='COM9'):
        self.ports = []
        self.connected = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        try:
            if self._check_ports(port_x, port_y):
                self.ports = [serial.Serial(port_x), serial.Serial(port_y)]
                self.connected = True
                print(f"Motors connected: {port_x}, {port_y}")
        except Exception as e:
            print(f"Motor connection error: {e}")
    
    def _check_ports(self, port_x, port_y):
        """Check if ports are available"""
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        return port_x in available_ports and port_y in available_ports
    
    def move(self, direction, step=None):
        """Move motors asynchronously"""
        if not self.connected:
            return
            
        if step is None:
            step_x = options['step_x']
            step_y = options['step_y']
        else:
            step_x = step_y = step
            
        def _move():
            try:
                if direction == 'r':
                    self.ports[0].write(f"M:1+P{step_x}\r\n".encode())
                    self.ports[0].write('G:\r\n'.encode())
                elif direction == 'l':
                    self.ports[0].write(f"M:1-P{step_x}\r\n".encode())
                    self.ports[0].write('G:\r\n'.encode())
                elif direction == 'u':
                    self.ports[1].write(f"M:1+P{step_y}\r\n".encode())
                    self.ports[1].write('G:\r\n'.encode())
                elif direction == 'd':
                    self.ports[1].write(f"M:1-P{step_y}\r\n".encode())
                    self.ports[1].write('G:\r\n'.encode())
                elif direction == 'o':
                    self.ports[0].write("H:1\r\n".encode())
                    self.ports[1].write("H:1\r\n".encode())
            except Exception as e:
                print(f"Motor move error: {e}")
        
        self.executor.submit(_move)
    
    def close(self):
        """Close motor connections"""
        self.executor.shutdown(wait=True)
        for port in self.ports:
            try:
                port.close()
            except:
                pass


class CustomWindow:
    """Custom window base class"""
    
    def __init__(self, *args, **kwargs):
        self.tk_title = "Arcy puszka"
        self.LGRAY = LGRAY
        self.DGRAY = DGRAY
        self.RGRAY = RGRAY
        self.MGRAY = MGRAY
        self._setup_window()
    
    def _setup_window(self):
        """Setup custom window elements"""
        self.title_bar = Frame(self, bg=self.RGRAY, relief='raised', bd=0, 
                              highlightthickness=1, highlightbackground=self.MGRAY)
        
        self.close_button = Button(self.title_bar, text='  ×  ', command=self.destroy, 
                                  bg=self.RGRAY, padx=2, pady=2, font=("calibri", 13), 
                                  bd=0, fg='lightgray', highlightthickness=0)
        
        self.minimize_button = Button(self.title_bar, text=' 🗕 ', command=self.minimize_me, 
                                     bg=self.RGRAY, padx=2, pady=2, bd=0, fg='lightgray', 
                                     font=("calibri", 13), highlightthickness=0)
        
        self.title_bar_title = Label(self.title_bar, text=self.tk_title, bg=self.RGRAY, 
                                    bd=0, fg='lightgray', font=("helvetica", 10))
        
        self.window = Frame(self, bg=self.DGRAY, highlightthickness=1, 
                           highlightbackground=self.MGRAY)
        
        # Pack elements
        self.title_bar.pack(fill=X)
        self.title_bar_title.pack(side=LEFT, padx=10)
        self.close_button.pack(side=RIGHT, ipadx=7, ipady=1)
        self.minimize_button.pack(side=RIGHT, ipadx=7, ipady=1)
        self.window.pack(expand=1, fill=BOTH)
        self.window.pack_propagate(1)

        # Bind events
        self.title_bar.bind('<Button-1>', self.get_pos)
        self.title_bar_title.bind('<Button-1>', self.get_pos)
        self.close_button.bind('<Enter>', lambda e: self.changex_on_hovering())
        self.close_button.bind('<Leave>', lambda e: self.returnx_to_normalstate())
        
        if hasattr(self, 'winfo_class') and self.winfo_class() == 'Tk':
            self.bind("<Expose>", lambda e: self.deminimize())
        self.after(10, lambda: self.set_appwindow())
    
    def get_pos(self, event):
        """Handle window dragging"""
        xwin = self.winfo_x()
        ywin = self.winfo_y()
        startx = event.x_root
        starty = event.y_root
        ywin = ywin - starty
        xwin = xwin - startx
        
        def move_window(event):
            self.config(cursor="fleur")
            self.geometry(f'+{event.x_root + xwin}+{event.y_root + ywin}')

        def release_window(event):
            self.config(cursor="arrow")

        self.title_bar.bind('<B1-Motion>', move_window)
        self.title_bar.bind('<ButtonRelease-1>', release_window)
    
    def set_appwindow(self):
        """Set window to appear in taskbar"""
        try:
            from ctypes import windll
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            hwnd = windll.user32.GetParent(self.winfo_id())
            stylew = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            stylew = stylew & ~WS_EX_TOOLWINDOW
            stylew = stylew | WS_EX_APPWINDOW
            windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, stylew)
            self.wm_withdraw()
            self.after(10, lambda: self.wm_deiconify())
        except:
            pass
    
    def minimize_me(self):
        """Minimize window"""
        self.overrideredirect(False)
        self.attributes('-alpha', 0)
        self.wm_state('iconic')
    
    def deminimize(self):
        """Restore window"""
        self.overrideredirect(True)
        self.attributes('-alpha', 1)
        self.wm_state('zoomed')
    
    def changex_on_hovering(self):
        """Close button hover effect"""
        self.close_button['bg'] = 'red'
    
    def returnx_to_normalstate(self):
        """Close button normal state"""
        self.close_button['bg'] = self.RGRAY
    
    def set_title(self, title):
        """Set window title"""
        self.tk_title = title
        self.title_bar_title.config(text=self.tk_title)
        if hasattr(self, 'title'):
            self.title(self.tk_title)


class CButton(Button):
    """Custom button with hover effects"""
    
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self.config(
            bg=RGRAY, padx=2, pady=2, bd=0, fg='lightgray',
            highlightthickness=0, relief='flat'
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<ButtonPress-1>', self.on_press)
        self.bind('<ButtonRelease-1>', self.on_release)

    def on_enter(self, event, color='gray'):
        self.config(bg=color)

    def on_leave(self, event):
        self.config(bg=RGRAY)

    def on_press(self, event):
        self.config(relief='sunken')

    def on_release(self, event):
        self.config(relief='flat')


class CustomTk(Tk, CustomWindow):
    """Custom main window"""
    
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.tk_title = "Spektrometr"
        self.overrideredirect(True)
        self.config(bg=self.DGRAY, highlightthickness=0)
        self.state('zoomed')


class CustomToplevel(Toplevel, CustomWindow):
    """Custom dialog window"""
    
    def __init__(self, *args, **kwargs):
        Toplevel.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.overrideredirect(True)
        self.config(bg=self.DGRAY, highlightthickness=0)


class HeatMapWindow(CustomToplevel):
    """3D heatmap visualization window"""
    
    def __init__(self, parent, measurement_index, data, image):
        CustomToplevel.__init__(self, parent)
        self.set_title(f'Measurement {measurement_index}')
        self.geometry('1200x800')
        
        self.data = np.array(data, dtype=object)
        self.parent = parent
        self._setup_data()
        self._create_widgets()
        self._update_plots()
    
    def _setup_data(self):
        """Setup data for visualization"""
        xs = sorted(set([row[0] for row in self.data]))
        ys = sorted(set([row[1] for row in self.data]))
        nx, ny = len(xs), len(ys)
        
        spectrum_len = len(self.data[0][2])
        self.cube = np.zeros((nx, ny, spectrum_len))
        
        for row in self.data:
            ix = xs.index(row[0])
            iy = ys.index(row[1])
            self.cube[ix, iy, :] = row[2]
        
        self.current_lambda = 0
        
        # Use calibration from options.json if available
        if hasattr(self.parent, 'options') and 'lambda_calibration_enabled' in self.parent.options:
            if self.parent.options['lambda_calibration_enabled']:
                lambda_min = self.parent.options.get('lambda_min', 400.0)
                lambda_max = self.parent.options.get('lambda_max', 700.0)
                self.lambdas = np.linspace(lambda_min, lambda_max, spectrum_len)
                self.calibrated = True
            else:
                # Use pixel-based calibration from xmin/xmax
                self.xmin = float(self.parent.xmin_var.get())
                self.xmax = float(self.parent.xmax_var.get())
                self.lambdas = np.linspace(self.xmin, self.xmax, spectrum_len)
                self.calibrated = False
        else:
            # Fallback to pixel-based calibration
            self.xmin = float(self.parent.xmin_var.get())
            self.xmax = float(self.parent.xmax_var.get())
            self.lambdas = np.linspace(self.xmin, self.xmax, spectrum_len)
            self.calibrated = False
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Main control frame at top
        main_control_frame = Frame(self.window, bg=self.DGRAY)
        main_control_frame.pack(fill=X, padx=10, pady=5)
        
        # Top row controls - left side for wavelength, right side for colormap
        top_row = Frame(main_control_frame, bg=self.DGRAY)
        top_row.pack(fill=X, pady=(0, 5))
        
        # Left side - Wavelength controls
        left_frame = Frame(top_row, bg=self.DGRAY)
        left_frame.pack(side=LEFT, fill=X, expand=True)
        
        # Wavelength label and value
        Label(left_frame, text="Wavelength:", bg=self.DGRAY, fg='white', font=('Arial', 10, 'bold')).pack(side=LEFT)
        self.wavelength_label = Label(left_frame, text="", bg=self.DGRAY, fg='lightgreen', font=('Arial', 10))
        self.wavelength_label.pack(side=LEFT, padx=(5,15))
        
        # Extended wavelength slider
        self.slider = Scale(
            left_frame, from_=0, to=self.cube.shape[2] - 1,
            orient=HORIZONTAL, command=self.on_slider,
            bg=self.DGRAY, fg='lightgray', length=500,  # Increased from 300 to 500
            highlightthickness=0, troughcolor=self.RGRAY,
            showvalue=False  # Hide the numeric labels above slider
        )
        self.slider.pack(side=LEFT, fill=X, expand=True, padx=(0,20))
        
        # Right side - Colormap selection
        right_frame = Frame(top_row, bg=self.DGRAY)
        right_frame.pack(side=RIGHT)
        
        Label(right_frame, text="Color Scale:", bg=self.DGRAY, fg='white', font=('Arial', 10, 'bold')).pack(side=LEFT, padx=(0,5))
        self.colormap_var = StringVar(value='hot')
        colormap_combo = ttk.Combobox(right_frame, textvariable=self.colormap_var, 
                                     values=['hot', 'viridis', 'plasma', 'cool', 'winter', 'autumn', 'spring', 'summer', 'gray', 'jet'],
                                     state='readonly', width=12)
        colormap_combo.pack(side=LEFT, padx=5)
        colormap_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plots())
        
        # Bottom row controls - calibration info and camera controls
        bottom_row = Frame(main_control_frame, bg=self.DGRAY)
        bottom_row.pack(fill=X, pady=(5, 0))
        
        # Calibration status
        cal_text = "Wavelength Calibrated" if self.calibrated else "Pixel Scale"
        cal_color = 'lightgreen' if self.calibrated else 'orange'
        Label(bottom_row, text="Scale:", bg=self.DGRAY, fg='white', font=('Arial', 9)).pack(side=LEFT)
        Label(bottom_row, text=cal_text, bg=self.DGRAY, fg=cal_color, font=('Arial', 9, 'bold')).pack(side=LEFT, padx=(5,20))
        
        # Create figure with 2D layout: 3D left, 2D right, spectrum bottom
        self.fig = plt.figure(figsize=(14, 10), facecolor=self.DGRAY)
        gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
        
        # 3D plot (top left)
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        
        # 2D heatmap (top right)
        self.ax2d = self.fig.add_subplot(gs[0, 1])
        
        # Spectrum plot (bottom row, spans both columns)
        self.ax_spectrum = self.fig.add_subplot(gs[1, :])
        
        # Initialize flags
        self.colorbar = None
        self._layout_set = False
        self._colorbar_created = False
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    def on_slider(self, val):
        """Handle slider change"""
        self.current_lambda = int(val)
        # Update wavelength label
        lambda_val = self.lambdas[self.current_lambda]
        unit = "nm" if self.calibrated else "px"
        self.wavelength_label.config(text=f"{lambda_val:.1f} {unit}")
        self._update_plots()
    
    def _update_plots(self):
        """Update 3D, 2D heatmap and spectrum plots"""
        try:
            # Get current colormap
            cmap = self.colormap_var.get()
            
            # Clear all plots
            self.ax3d.clear()
            self.ax2d.clear()
            self.ax_spectrum.clear()
            
            lambda_val = self.lambdas[self.current_lambda]
            unit = "nm" if self.calibrated else "px"
            
            # 3D heatmap (top left)
            data = self.cube[:, :, self.current_lambda]
            X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
            surf = self.ax3d.plot_surface(X, Y, data.T, cmap=cmap, alpha=0.8)
            
            self.ax3d.set_title(f"3D Heatmap - λ={lambda_val:.1f} {unit}", color='white', fontsize=12)
            self.ax3d.set_xlabel("X Position", color='white')
            self.ax3d.set_ylabel("Y Position", color='white')
            self.ax3d.set_zlabel("Intensity", color='white')
            
            # 2D heatmap (top right)
            im = self.ax2d.imshow(data.T, cmap=cmap, aspect='auto', origin='lower',
                                extent=[0, data.shape[0], 0, data.shape[1]], 
                                interpolation='nearest')
            self.ax2d.set_title(f"2D Heatmap - λ={lambda_val:.1f} {unit}", color='white', fontsize=12)
            self.ax2d.set_xlabel("X Position", color='white')
            self.ax2d.set_ylabel("Y Position", color='white')
            
            # Set fixed layout
            if not self._layout_set:
                self.fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, 
                                       wspace=0.3, hspace=0.3)
                self._layout_set = True
            
            # Create/update colorbar for 2D heatmap
            if not hasattr(self, '_colorbar_created') or not self._colorbar_created:
                try:
                    self.colorbar = self.fig.colorbar(im, ax=self.ax2d, fraction=0.046, 
                                                    pad=0.04, shrink=0.8)
                    self.colorbar.ax.tick_params(colors='white')
                    self._colorbar_created = True
                except Exception as e:
                    print(f"Error creating colorbar: {e}")
                    self._colorbar_created = False
            else:
                try:
                    self.colorbar.update_normal(im)
                except Exception:
                    try:
                        self.colorbar.remove()
                        self.colorbar = self.fig.colorbar(im, ax=self.ax2d, fraction=0.046, 
                                                        pad=0.04, shrink=0.8)
                        self.colorbar.ax.tick_params(colors='white')
                    except Exception as e:
                        print(f"Error recreating colorbar: {e}")
            
            # Spectrum plot (bottom, full width)
            mean_profile = self.cube.mean(axis=(0, 1))
            self.ax_spectrum.plot(self.lambdas, mean_profile, color='orange', linewidth=2, 
                                label="Average Spectrum", alpha=0.8)
            self.ax_spectrum.axvline(lambda_val, color='red', linestyle='--', linewidth=2, 
                                   label=f"Current λ={lambda_val:.1f} {unit}")
            
            # Add wavelength range info if calibrated
            if self.calibrated:
                self.ax_spectrum.set_title("Calibrated Spectrum Profile", color='white', fontsize=14)
                self.ax_spectrum.set_xlabel("Wavelength (nm)", color='white')
            else:
                self.ax_spectrum.set_title("Spectrum Profile (Pixel Scale)", color='white', fontsize=14)
                self.ax_spectrum.set_xlabel("Pixel Position", color='white')
            
            self.ax_spectrum.set_ylabel("Intensity", color='white')
            self.ax_spectrum.legend(facecolor=self.DGRAY, edgecolor='white', 
                                  labelcolor='white', fontsize=10)
            self.ax_spectrum.grid(True, alpha=0.3, color='gray')
            
            # Style all plots
            for ax in [self.ax3d, self.ax2d, self.ax_spectrum]:
                ax.set_facecolor(self.DGRAY)
                ax.tick_params(colors='white')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()
            print(f"Error updating plots: {e}")
            # Try to continue with basic plot update
            try:
                self.canvas.draw()
            except:
                pass


class SpektrometerApp(CustomTk):
    """Main application class"""
    
    def __init__(self):
        super().__init__()
        self.title("Spektrometr")
        self.geometry('1400x900')
        
        # Store options reference for access from other components
        self.options = options
        
        # Selected camera index from options
        self.camera_index = int(options.get('camera_index', 0))

        # Initialize managers
        self.camera_manager = CameraManager(camera_index=self.camera_index)
        self.spectrometer_manager = SpectrometerManager()
        self.motor_controller = MotorController(
            options.get('port_x', 'COM10'),
            options.get('port_y', 'COM11')
        )
        
        # Variables
        self.measurement_files = []  # Store filenames only, not data
        self.current_image = None
        self.spectrum_data = np.zeros(2048)
        self.cal_px_per_step_x = float(options.get('cal_px_per_step_x', 0.0))
        self.cal_px_per_step_y = float(options.get('cal_px_per_step_y', 0.0))
        self.cal_sign_x = int(options.get('cal_sign_x', 1))
        self.cal_sign_y = int(options.get('cal_sign_y', 1))
        
        # Sequence control flags
        self._sequence_running = False
        self._sequence_stop_requested = False
        
        # Calibration state: always start as False, must be performed in current session
        self.calibration = False
        
        # Default spectrum range variables
        self.xmin_var = StringVar(value=options.get('xmin', '0'))
        self.xmax_var = StringVar(value=options.get('xmax', '2048'))
        
        # Background threads will be started in _delayed_init
        
        # Setup GUI FIRST before redirecting stdout
        self._create_widgets()
        self._setup_styles()
        
        # NOW redirect stdout after console is created
        sys.stdout = StreamToFunction(self.console_output)
        
        # Initialize systems and start update loop
        self.after(100, self._delayed_init)

    def _create_widgets(self):
        """Create main GUI widgets"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs - COMBINED TABS
        self.tab_camera_controls = Frame(self.notebook, bg=self.DGRAY)  # Camera + Controls
        self.tab_spectrum_pixelink = Frame(self.notebook, bg=self.DGRAY)  # Spectrum + Pixelink
        self.tab_results = Frame(self.notebook, bg=self.DGRAY)
        self.tab_settings = Frame(self.notebook, bg=self.DGRAY)
        
        self.notebook.add(self.tab_camera_controls, text="Camera & Controls")
        self.notebook.add(self.tab_spectrum_pixelink, text="Spectrum")
        self.notebook.add(self.tab_results, text="Results")
        self.notebook.add(self.tab_settings, text="Settings")
        
        # Setup individual tabs
        self._setup_camera_controls_tab()
        self._setup_spectrum_pixelink_tab()
        self._setup_results_tab()
        self._setup_settings_tab()

    def _setup_camera_controls_tab(self):
        """Setup simplified camera tab with centered view and calibrate button only"""
        # Create main container 
        main_container = Frame(self.tab_camera_controls, bg=self.DGRAY)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Camera title
        Label(main_container, text="Camera", font=("Arial", 16, "bold"), 
              bg=self.DGRAY, fg='white').pack(pady=(0, 10))
        
        # Centered camera display (Canvas with drag select) - moderate size
        camera_container = Frame(main_container, bg=self.DGRAY)
        camera_container.pack(expand=False, fill=X)
        
        self.camera_canvas = Canvas(camera_container, bg=self.DGRAY, highlightthickness=0)
        self.camera_canvas.pack()
        self._camera_canvas_img = None  # Initialize as None instead of reading from camera
        self._camera_frame_size = (800, 600)  # Moderate camera view to leave space for controls
        self.camera_canvas.config(width=self._camera_frame_size[0], height=self._camera_frame_size[1])
        
        # Draw placeholder if no camera
        
        # Status under canvas
        self.cam_status = Label(main_container, bg=self.DGRAY, fg='lightgray', 
                               text="Camera Status: Not Started", font=("Arial", 10))
        self.cam_status.pack(pady=5)
        
        # Control buttons frame - centered
        control_frame = Frame(main_container, bg=self.DGRAY)
        control_frame.pack(pady=10)
        
        # Essential controls only
        CButton(control_frame, text="Start Camera", command=lambda: self.start_camera()).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Stop Camera", command=lambda: self.stop_camera()).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Calibrate", command=self.start_calibration_both).pack(side=LEFT, padx=10)
        
        # Add Start Sequence button (needed for state management)
        self.start_seq_btn = CButton(control_frame, text="Start Sequence", command=self.start_measurement_sequence)
        self.start_seq_btn.pack(side=LEFT, padx=5)
        
        # Add Stop Sequence button
        self.stop_seq_btn = CButton(control_frame, text="Stop Sequence", command=self.stop_measurement_sequence)
        self.stop_seq_btn.pack(side=LEFT, padx=5)
        self.stop_seq_btn.config(state=DISABLED)  # Initially disabled
        
        # Initial state based on calibration
        self._update_start_seq_state()
        
        # Motor control section
        motor_frame = LabelFrame(main_container, text="Manual Motor Control", bg=self.DGRAY, fg='white')
        motor_frame.pack(fill=X, pady=10)
        
        # Horizontal layout: Console on left, buttons on right
        horizontal_frame = Frame(motor_frame, bg=self.DGRAY)
        horizontal_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Console (larger)
        console_frame = LabelFrame(horizontal_frame, text="Status Console", bg=self.DGRAY, fg='white')
        console_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        self.console = Text(console_frame, bg=self.DGRAY, fg='lightgreen', height=10, wrap=WORD)
        console_scrollbar = Scrollbar(console_frame, orient=VERTICAL, command=self.console.yview)
        self.console.configure(yscrollcommand=console_scrollbar.set)
        
        # Configure color tags for console
        self.console.tag_configure("error", foreground="red")
        self.console.tag_configure("warning", foreground="yellow")
        self.console.tag_configure("normal", foreground="lightgreen")
        
        self.console.pack(side=LEFT, fill=BOTH, expand=True)
        console_scrollbar.pack(side=RIGHT, fill=Y)
        
        # Right side - Motor controls (smaller)
        motor_controls_frame = Frame(horizontal_frame, bg=self.DGRAY)
        motor_controls_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))
        
        # Step size controls
        Label(motor_controls_frame, text="Step Size:", bg=self.DGRAY, fg='white', font=("Arial", 10)).pack(pady=(0, 5))
        step_control_frame = Frame(motor_controls_frame, bg=self.DGRAY)
        step_control_frame.pack(pady=(0, 10))
        
        self.motor_step_var = IntVar(value=1)
        step_sizes = [1, 5, 10, 25, 50]
        for i, step in enumerate(step_sizes):
            if i < 3:  # First row
                row, col = 0, i
            else:  # Second row
                row, col = 1, i-3
            Radiobutton(step_control_frame, text=str(step), variable=self.motor_step_var, value=step,
                       bg=self.DGRAY, fg='white', selectcolor=self.RGRAY, font=("Arial", 9),
                       activebackground=self.RGRAY).grid(row=row, column=col, padx=1, pady=1)
        
        # Directional buttons in cross pattern (smaller)
        Label(motor_controls_frame, text="Movement:", bg=self.DGRAY, fg='white', font=("Arial", 10)).pack(pady=(10, 5))
        direction_frame = Frame(motor_controls_frame, bg=self.DGRAY)
        direction_frame.pack()
        
        # Row 1: Up button
        Button(direction_frame, text="↑", command=lambda: self.move_motor('u'), 
               width=4, bg=RGRAY, fg='white', font=("Arial", 10)).grid(row=0, column=1, padx=2, pady=2)
        
        # Row 2: Left, Origin, Right buttons  
        Button(direction_frame, text="←", command=lambda: self.move_motor('l'), 
               width=4, bg=RGRAY, fg='white', font=("Arial", 10)).grid(row=1, column=0, padx=2, pady=2)
        Button(direction_frame, text="⌂", command=lambda: self.move_motor('o'), 
               width=4, bg=RGRAY, fg='white', font=("Arial", 10)).grid(row=1, column=1, padx=2, pady=2)
        Button(direction_frame, text="→", command=lambda: self.move_motor('r'), 
               width=4, bg=RGRAY, fg='white', font=("Arial", 10)).grid(row=1, column=2, padx=2, pady=2)
        
        # Row 3: Down button
        Button(direction_frame, text="↓", command=lambda: self.move_motor('d'), 
               width=4, bg=RGRAY, fg='white', font=("Arial", 10)).grid(row=2, column=1, padx=2, pady=2)
        
        # Motor status
        self.motor_status = Label(motor_controls_frame, bg=self.DGRAY, fg='lightgray', 
                                 text="Motor Status: Ready", font=("Arial", 9), wraplength=150)
        self.motor_status.pack(pady=(10, 0))
        
    def start_camera(self):
        """Start camera preview and enable live camera in heatmaps"""
        if self.camera_manager and not self.camera_manager.running:
            success = self.camera_manager.start()
            if success:
                # Initialize placeholder instead of frame_buffer (which doesn't exist)
                self._camera_canvas_img = None
                self.cam_status.config(text="Camera started - Live view active", fg='lightgreen')
                
            else:
                self.cam_status.config(text="Failed to start camera", fg='red')
        else:
            print("Camera already running or not initialized")

    def stop_camera(self):
        """Stop camera preview"""
        if self.camera_manager and self.camera_manager.running:
            self.camera_manager.stop()
            self.cam_status.config(text="Camera stopped", fg='orange')
            print("Camera stopped")
        else:
            print("Camera is not running")

    def _setup_spectrum_pixelink_tab(self):
        """Setup spectrum tab with image preview and spectrum plot"""
        # Create main container
        main_container = Frame(self.tab_spectrum_pixelink, bg=self.DGRAY)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Image Preview (60% of space)
        image_frame = Frame(main_container, bg=self.DGRAY)
        image_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # Image title and status
        image_header = Frame(image_frame, bg=self.DGRAY)
        image_header.pack(fill=X, pady=(0, 5))
        
        Label(image_header, text="PixeLink Camera", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack(side=LEFT)
        
        self.pixelink_status = Label(
            image_header,
            text="🔄 Initializing...",
            bg=self.DGRAY, fg='yellow', font=("Arial", 10)
        )
        self.pixelink_status.pack(side=RIGHT)
        
        # Simple image display
        self.spectrum_live_label = Label(
            image_frame,
            bg='black',
            text="Camera Preview\nInitializing...",
            fg='white',
            font=("Arial", 12),
            justify=CENTER
        )
        self.spectrum_live_label.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Bottom section - Spectrum Plot (40% of space) 
        spectrum_frame = Frame(main_container, bg=self.DGRAY)
        spectrum_frame.pack(fill=X, pady=(5, 0))
        spectrum_frame.pack_propagate(False)
        spectrum_frame.configure(height=200)
        
        # Create spectrum plot directly
        self.spectrum_fig, self.spectrum_ax = plt.subplots(figsize=(10, 3), facecolor=self.DGRAY)
        self.spectrum_ax.set_facecolor(self.DGRAY)
        
        self.x_axis = np.linspace(0, 2048, 2048)
        self.spectrum_data = np.zeros(2048)
        self.spectrum_line, = self.spectrum_ax.plot(self.x_axis, self.spectrum_data, color='green', linewidth=1)
        
        # Style
        self.spectrum_ax.set_xlabel("Pixel", color='white', fontsize=10)
        self.spectrum_ax.set_ylabel("Intensity", color='white', fontsize=10)
        self.spectrum_ax.set_title("Spectrum", color='white', fontsize=12)
        self.spectrum_ax.tick_params(colors='white', labelsize=8)
        self.spectrum_ax.grid(True, alpha=0.3, color='gray')
        
        self.spectrum_fig.tight_layout()
        
        # Canvas
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=spectrum_frame)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Start unified update thread for both camera and spectrometer
        def update_image():
            def update_camera_display(frame):
                """Update camera display - inline function"""
                try:
                    if frame is None or frame.size == 0:
                        return
                        
                    # Resize for camera canvas display
                    height, width = frame.shape[:2]
                    canvas_w, canvas_h = self._camera_frame_size
                    
                    # Scale to fit canvas
                    scale = min(canvas_w/width, canvas_h/height)
                    new_w, new_h = int(width*scale), int(height*scale)
                    
                    if new_w > 0 and new_h > 0:
                        frame_resized = cv2.resize(frame, (new_w, new_h))
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        
                        # Create PIL image and PhotoImage
                        pil_image = Image.fromarray(frame_rgb)
                        photo = ImageTk.PhotoImage(pil_image)
                        
                        # Update camera canvas
                        if hasattr(self, 'camera_canvas'):
                            self.camera_canvas.delete("all")
                            x_offset = (canvas_w - new_w) // 2
                            y_offset = (canvas_h - new_h) // 2
                            self.camera_canvas.create_image(x_offset, y_offset, anchor='nw', image=photo)
                            self.camera_canvas.image = photo  # Keep reference
                            
                        # Update status
                        if hasattr(self, 'cam_status'):
                            self.cam_status.config(text="Camera: Live feed active", fg='lightgreen')
                    
                except Exception as e:
                    print(f"Camera display error: {e}")
            
            def update_spectrum_display(frame):
                """Update spectrometer image display - inline function"""
                try:
                    if frame is None or frame.size == 0:
                        return
                        
                    # Create PIL image from buffer
                    pil_image = Image.fromarray(frame.copy())
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    self.spectrum_live_label.configure(image=photo, text="")
                    self.spectrum_live_label.image = photo
                    
                    # Update status
                    self.pixelink_status.configure(text="🟢 Live", fg='lightgreen')
                    
                except Exception as e:
                    print(f"Spectrum display error: {e}")
            
            while not getattr(self, '_stop_threads', False):
                try:
                    # Update camera display using direct method
                    if (hasattr(self, 'camera_manager') and 
                        self.camera_manager and 
                        self.camera_manager.running):
                        
                        camera_frame = self.camera_manager.get_current_frame()
                        if camera_frame is not None:
                            # Update camera display in main thread
                            self.after_idle(lambda f=camera_frame: update_camera_display(f))
                    
                    # Update spectrometer display using direct buffer access
                    if (hasattr(self, 'spectrometer_manager') and 
                        self.spectrometer_manager and 
                        self.spectrometer_manager.running and
                        hasattr(self.spectrometer_manager, 'frame_buffer')):
                        
                        frame_buffer = self.spectrometer_manager.frame_buffer
                        if frame_buffer is not None and frame_buffer.size > 0:
                            # Update spectrum display in main thread
                            self.after_idle(lambda f=frame_buffer: update_spectrum_display(f))
                    
                    time.sleep(0.03)  # 30 FPS max for both sources
                except Exception as e:
                    print(f"Update thread error: {e}")
                    time.sleep(0.1)
        
        # Start unified thread
        threading.Thread(target=update_image, daemon=True).start()

    def _update_start_seq_state(self):
        try:
            # Check conditions - calibration must be completed and not in progress
            cal_status = getattr(self, 'calibration', False)
            cal_in_progress = getattr(self, '_calibration_mode', None) is not None
            
            # Enable only if calibrated AND not currently calibrating
            enabled = bool(cal_status and not cal_in_progress)
            
            if hasattr(self, 'start_seq_btn'):
                old_state = str(self.start_seq_btn.cget('state'))
                self.start_seq_btn.configure(state=NORMAL if enabled else DISABLED)
                
                # Update button appearance and text based on state
                if enabled:
                    self.start_seq_btn.configure(
                        text="START SEQUENCE", 
                        bg='#28a745',  # Green background when enabled
                        fg='white'
                    )
                    # Print message when sequence button becomes enabled
                    if old_state == 'disabled':
                        print("SEQUENCE BUTTON UNLOCKED - Ready to start measurements!")
                        print("Click 'START SEQUENCE' to begin automatic scanning")
                else:
                    # Determine why it's disabled
                    if cal_in_progress:
                        button_text = "Start Sequence (calibrating...)"
                    elif not cal_status:
                        button_text = "Start Sequence (no calibration)"
                    else:
                        button_text = "Start Sequence (disabled)"
                        
                    self.start_seq_btn.configure(
                        text=button_text, 
                        bg='#6c757d',  # Gray background when disabled
                        fg='lightgray'
                    )
        except Exception as e:
            print(f"Error updating button state: {e}")


    def _setup_results_tab(self):
        """Setup results tab"""
        # Control buttons at top
        control_frame = Frame(self.tab_results, bg=self.DGRAY)
        control_frame.pack(fill=X, padx=5, pady=5)
        
        CButton(control_frame, text="Refresh", command=self.load_measurements).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Export All", command=self.export_measurements).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Delete All", command=self.delete_all_measurements).pack(side=LEFT, padx=5)
        
        # Info label
        self.results_info = Label(
            control_frame, 
            text="Measurements: 0", 
            bg=self.DGRAY, fg='lightgray'
        )
        self.results_info.pack(side=RIGHT, padx=10)
        
        # Main frame with canvas for scrolling
        main_frame = Frame(self.tab_results, bg=self.DGRAY)
        main_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Canvas and scrollbars
        self.results_canvas = Canvas(main_frame, bg=self.DGRAY, highlightthickness=0)
        v_scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=self.results_canvas.yview)
        h_scrollbar = Scrollbar(main_frame, orient=HORIZONTAL, command=self.results_canvas.xview)
        
        # Scrollable frame inside canvas
        self.results_frame = Frame(self.results_canvas, bg=self.DGRAY)
        
        # Configure scrolling
        self.results_canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Create window in canvas
        self.canvas_frame = self.results_canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw"
        )
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=RIGHT, fill=Y)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        self.results_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Bind canvas resize
        self.results_canvas.bind('<Configure>', self._on_canvas_configure)
        self.results_frame.bind('<Configure>', self._on_frame_configure)
        
        # Bind mousewheel to canvas
        self.results_canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _setup_settings_tab(self):
        """Setup settings tab"""
        settings_frame = Frame(self.tab_settings, bg=self.DGRAY)
        settings_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Movement settings
        Label(settings_frame, text="Movement Settings", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=0, column=0, columnspan=2, pady=10)
        
        # Variables for settings
        self.step_x = IntVar(value=options.get('step_x', 10))
        self.step_y = IntVar(value=options.get('step_y', 10))
        self.scan_width = IntVar(value=options.get('width', 100))
        self.scan_height = IntVar(value=options.get('height', 100))
        
        # Settings entries
        settings_data = [
            ("Step X:", self.step_x),
            ("Step Y:", self.step_y),
            ("Scan Width:", self.scan_width),
            ("Scan Height:", self.scan_height),
        ]
        
        for i, (label, var) in enumerate(settings_data, 1):
            Label(settings_frame, text=label, bg=self.DGRAY, fg='white').grid(row=i, column=0, sticky=W, pady=5)
            Entry(settings_frame, textvariable=var, bg=self.RGRAY, fg='white').grid(row=i, column=1, sticky=EW, pady=5)

        # Port settings only (moved camera + calibration to Camera & Controls)
        row_base = len(settings_data) + 2
        Label(settings_frame, text="Port Settings", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=row_base, column=0, columnspan=3, pady=10, sticky=W)

        ports = [p.device for p in serial.tools.list_ports.comports()]

        Label(settings_frame, text="Port X:", bg=self.DGRAY, fg='white').grid(row=row_base+1, column=0, sticky=W, pady=5)
        self.port_x_var = StringVar(value=options.get('port_x', 'COM5'))
        self.port_x_combo = ttk.Combobox(settings_frame, textvariable=self.port_x_var, values=ports)
        self.port_x_combo.grid(row=row_base+1, column=1, sticky=EW, pady=5)

        Label(settings_frame, text="Port Y:", bg=self.DGRAY, fg='white').grid(row=row_base+2, column=0, sticky=W, pady=5)
        self.port_y_var = StringVar(value=options.get('port_y', 'COM9'))
        self.port_y_combo = ttk.Combobox(settings_frame, textvariable=self.port_y_var, values=ports)
        self.port_y_combo.grid(row=row_base+2, column=1, sticky=EW, pady=5)

        # Sequence sleep setting
        Label(settings_frame, text="Sequence Sleep (s):", bg=self.DGRAY, fg='white').grid(row=row_base+3, column=0, sticky=W, pady=5)
        self.sequence_sleep_var = DoubleVar(value=options.get('sequence_sleep', 0.1))
        sequence_sleep_entry = Entry(settings_frame, textvariable=self.sequence_sleep_var, bg=self.MGRAY, fg='white')
        sequence_sleep_entry.grid(row=row_base+3, column=1, sticky=EW, pady=5)

        CButton(settings_frame, text="Refresh Ports", command=self.refresh_ports).grid(row=row_base+1, column=2, rowspan=3, padx=10, sticky=N)

        # Camera settings
        cam_row = row_base + 4
        Label(settings_frame, text="Camera Settings", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=cam_row, column=0, columnspan=3, pady=10, sticky=W)
        
        Label(settings_frame, text="Camera Index:", bg=self.DGRAY, fg='white').grid(row=cam_row+1, column=0, sticky=W, pady=5)
        self.camera_index_var = IntVar(value=options.get('camera_index', 0))
        cams = [0, 1, 2]  # self._list_cameras() # REMOVED TEMPORARILY
        self.camera_combo = ttk.Combobox(settings_frame, values=cams, state='readonly', width=10)
        try:
            self.camera_combo.set(str(self.camera_index_var.get()) if self.camera_index_var.get() in cams else str(cams[0]))
        except Exception:
            pass
        self.camera_combo.grid(row=cam_row+1, column=1, sticky=EW, pady=5)
        CButton(settings_frame, text="Refresh", command=lambda: print("Camera refresh disabled")).grid(row=cam_row+1, column=2, padx=10)
        
        # Calibration settings
        calib_row = cam_row + 2
        Label(settings_frame, text="Calibration (px/step)", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=calib_row, column=0, columnspan=3, pady=10, sticky=W)
        
        Label(settings_frame, text="X:", bg=self.DGRAY, fg='white').grid(row=calib_row+1, column=0, sticky=W, pady=5)
        self.cal_x_var = DoubleVar(value=self.cal_px_per_step_x)
        Entry(settings_frame, textvariable=self.cal_x_var, bg=self.RGRAY, fg='white').grid(row=calib_row+1, column=1, sticky=EW, pady=5)
        
        Label(settings_frame, text="Y:", bg=self.DGRAY, fg='white').grid(row=calib_row+2, column=0, sticky=W, pady=5)
        self.cal_y_var = DoubleVar(value=self.cal_px_per_step_y)
        Entry(settings_frame, textvariable=self.cal_y_var, bg=self.RGRAY, fg='white').grid(row=calib_row+2, column=1, sticky=EW, pady=5)

        # Wavelength calibration settings
        wave_row = calib_row + 3
        Label(settings_frame, text="Wavelength Calibration", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=wave_row, column=0, columnspan=3, pady=10, sticky=W)
        
        # Calibration enable checkbox
        self.lambda_cal_enabled_var = BooleanVar(value=options.get('lambda_calibration_enabled', True))
        lambda_enable_check = Checkbutton(settings_frame, text="Enable Wavelength Calibration", 
                                        variable=self.lambda_cal_enabled_var,
                                        bg=self.DGRAY, fg='white', selectcolor=self.RGRAY,
                                        activebackground=self.DGRAY, activeforeground='lightgreen',
                                        font=('Arial', 10))
        lambda_enable_check.grid(row=wave_row+1, column=0, columnspan=2, sticky=W, pady=5)
        
        # Lambda min
        Label(settings_frame, text="λ min (nm):", bg=self.DGRAY, fg='white').grid(row=wave_row+2, column=0, sticky=W, pady=5)
        self.lambda_min_var = DoubleVar(value=options.get('lambda_min', 400.0))
        Entry(settings_frame, textvariable=self.lambda_min_var, bg=self.RGRAY, fg='white').grid(row=wave_row+2, column=1, sticky=EW, pady=5)
        
        # Lambda max
        Label(settings_frame, text="λ max (nm):", bg=self.DGRAY, fg='white').grid(row=wave_row+3, column=0, sticky=W, pady=5)
        self.lambda_max_var = DoubleVar(value=options.get('lambda_max', 700.0))
        Entry(settings_frame, textvariable=self.lambda_max_var, bg=self.RGRAY, fg='white').grid(row=wave_row+3, column=1, sticky=EW, pady=5)
        
        # Info label
        Label(settings_frame, text="Note: Wavelength calibration will override pixel scale in heatmaps", 
              bg=self.DGRAY, fg='orange', font=("Arial", 9, "italic")).grid(row=wave_row+4, column=0, columnspan=3, sticky=W, pady=5)

        # Apply button - make it more prominent
        apply_frame = Frame(settings_frame, bg=self.DGRAY)
        apply_frame.grid(row=wave_row+6, column=0, columnspan=3, pady=20)
        
        CButton(apply_frame, text="SAVE SETTINGS", command=self.apply_settings, 
               font=("Arial", 12, "bold"), fg='yellow').pack(pady=5)
        
        Label(apply_frame, text="Click to save all changes to options.json", 
              bg=self.DGRAY, fg='lightgray', font=("Arial", 9)).pack()
        
        settings_frame.columnconfigure(1, weight=1)

    def refresh_ports(self):
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            for combo in [self.port_x_combo, self.port_y_combo]:
                combo.configure(values=ports)
            print(f"Ports refreshed: {ports}")
        except Exception as e:
            print(f"Ports refresh error: {e}")

    def save_calibration(self):
        try:
            self.cal_px_per_step_x = float(self.cal_x_var.get())
            self.cal_px_per_step_y = float(self.cal_y_var.get())
            options['cal_px_per_step_x'] = self.cal_px_per_step_x
            options['cal_px_per_step_y'] = self.cal_px_per_step_y
            options['cal_sign_x'] = int(getattr(self, 'cal_sign_x', 1))
            options['cal_sign_y'] = int(getattr(self, 'cal_sign_y', 1))
            # Keep existing cal_step values (from options/settings)
            options['cal_step_x'] = int(options.get('cal_step_x', 1))
            options['cal_step_y'] = int(options.get('cal_step_y', 1))
            with open('options.json', 'w') as f:
                json.dump(options, f, indent=4)
            print(f"Calibration saved: X={self.cal_px_per_step_x:.4f}, Y={self.cal_px_per_step_y:.4f}")
        except Exception as e:
            print(f"Calibration save error: {e}")

    def _setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.DGRAY, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.DGRAY, foreground='white')
        style.map('TNotebook.Tab', background=[('selected', self.RGRAY)])

    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        canvas_width = event.width
        self.results_canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def _on_frame_configure(self, event):
        """Handle frame resize"""
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def console_output(self, message):
        """Output text to console Text widget with color coding"""
        try:
            if not hasattr(self, 'console') or not self.console.winfo_exists():
                return
            readable_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            full_message = f'{readable_time}: {message}\n'
            
            # Detect error messages and apply red color
            is_error = any(keyword in message.lower() for keyword in [
                'error', 'error', 'exception', 'failed', 'failed', 'not connected',
                'check connection', 'traceback', 'not detected'
            ])
            
            # Detect warning messages and apply yellow color
            is_warning = any(keyword in message.lower() for keyword in [
                'warning', 'warning', 'attention', 'attention', 'simulation', 'simulation',
                'not connected', 'not connected', 'approximate', 'approximate'
            ])
            
            # Choose appropriate tag
            if is_error:
                tag = "error"
            elif is_warning:
                tag = "warning"
            else:
                tag = "normal"
            
            self.console.insert(END, full_message, tag)
            self.console.see(END)
        except Exception:
            pass

    def _delayed_init(self):
        """Initialize systems after GUI is ready - move heavy operations to background"""
        print("Starting system initialization...")
        
        # Start background initialization thread to avoid freezing UI
        init_thread = threading.Thread(target=self._background_initialization, daemon=True)
        init_thread.start()
        
        print("GUI ready - background initialization in progress...")

    def _background_initialization(self):
        """Heavy initialization operations in background thread"""
        try:
            print("Background: Initializing spectrometer...")
            # Initialize spectrometer in background
            if self.spectrometer_manager.initialize():
                self.spectrometer_manager.start()
                self.after_idle(lambda: print("Spectrometer initialized successfully"))
            else:
                self.after_idle(lambda: print("Spectrometer initialization failed"))
                # Do not auto-load any test image; require real camera frames
            
            print("Background: Loading measurements...")
            # Load measurements in background
            self.load_measurements()
            
            # Auto-initialize PixeLink camera
            print("Background: Auto-initializing PixeLink...")
            self.after_idle(self._auto_init_pixelink)
            
            self.after_idle(lambda: print("✅ System initialization complete"))
            
        except Exception as e:
            self.after_idle(lambda: print(f"❌ Background initialization error: {e}"))

    def _auto_init_pixelink(self):
        """Auto-initialize PixeLink camera without user interaction"""
        try:
            print("🔄 Auto-initializing PixeLink camera...")
            
            # Initialize PixeLink
            if self.init_pixelink():
                print("✅ PixeLink auto-initialized successfully")
                
                # Auto-start streaming after short delay
                self.after(1000, self._auto_start_pixelink)  # 1 second delay
            else:
                print("❌ PixeLink auto-initialization failed")
                
        except Exception as e:
            print(f"❌ PixeLink auto-init error: {e}")
    
    def _auto_start_pixelink(self):
        """Auto-start PixeLink streaming"""
        try:
            print("🔄 Auto-starting PixeLink stream...")
            self.start_pixelink()
            print("✅ PixeLink stream auto-started")
        except Exception as e:
            print(f"❌ PixeLink auto-start error: {e}")

    # REMOVED: _init_display_placeholders() - function removed temporarily

    # REMOVED: _fallback_to_test_image() - function removed temporarily

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        self._stop_threads = True
        try:
            if hasattr(self, 'camera_manager'):
                self.camera_manager.stop()
            if hasattr(self, 'spectrometer_manager'):
                self.spectrometer_manager.stop()
            if hasattr(self, 'motor_controller'):
                self.motor_controller.close()
        except Exception as e:
            print(f"Cleanup error: {e}")
        # Restore stdout
        try:
            sys.stdout = sys.__stdout__
        except Exception:
            pass

    # REMOVED: start_camera() - function removed temporarily
                
    # REMOVED: stop_camera() - function removed temporarily

    def _move_and_record(self, direction, steps=None):
        try:
            self._last_move_dir = direction
            self.motor_controller.move(direction, steps)
        except Exception as e:
            print(f"Move error: {e}")

    # REMOVED: Mouse interaction functions - not used in simple spectrum view

    # REMOVED: Spectrum control functions - not used in simple view

    def _update_spectrum_plot(self):
        """Update spectrum plot with enhanced features - completely non-blocking"""
        try:
            if not hasattr(self, 'spectrum_data') or len(self.spectrum_data) == 0:
                return
                
            # Update main spectrum line
            if hasattr(self, 'spectrum_line'):
                self.spectrum_line.set_ydata(self.spectrum_data)
                
                # Auto-adjust Y axis if not manually zoomed
                current_ylim = self.spectrum_ax.get_ylim()
                current_xlim = self.spectrum_ax.get_xlim()
                
                # Check if we need to update Y scale
                data_max = np.max(self.spectrum_data)
                data_min = np.min(self.spectrum_data)
                
                # Only auto-scale if the current view shows the full range
                if (current_xlim[0] <= 0 and current_xlim[1] >= len(self.spectrum_data) - 1):
                    if data_max > current_ylim[1] * 0.9 or data_max < current_ylim[1] * 0.5:
                        # Prevent identical ylim values which cause matplotlib warnings
                        if abs(data_max - data_min) < 1e-10:  # Values are essentially the same
                            y_center = data_min if data_min != 0 else 1
                            self.spectrum_ax.set_ylim(y_center - abs(y_center) * 0.1, y_center + abs(y_center) * 0.1)
                        else:
                            self.spectrum_ax.set_ylim(data_min * 0.9, data_max * 1.1)
                
                # Safe canvas update in main thread - inline
                def safe_canvas_update():
                    try:
                        if hasattr(self, 'spectrum_canvas'):
                            self.spectrum_canvas.draw_idle()
                    except Exception:
                        pass  # Ignore canvas errors to prevent freezing
                
                self.after_idle(safe_canvas_update)
                
        except Exception as e:
            print(f"Spectrum plot update error: {e}")

    # REMOVED: load_test_image() - function removed temporarily

    def init_pixelink(self):
        """Initialize Pixelink camera with enhanced status updates"""
        try:
            print("=== PIXELINK INITIALIZATION ===")
            
            # Check if Pixelink API is available
            try:
                print("✓ Pixelink API imported successfully")
            except ImportError as e:
                print(f"✗ Pixelink API import failed: {e}")
                self.pixelink_status.config(text="❌ API Missing", fg='red')
                if hasattr(self, 'live_indicator'):
                    self.live_indicator.configure(text="❌ API Missing", fg='red')
                return False
                
            if self.spectrometer_manager.initialize():
                self.pixelink_status.config(text="✅ Initialized", fg='lightgreen')
                if hasattr(self, 'live_indicator'):
                    self.live_indicator.configure(text="✅ Ready", fg='yellow')
                print("✓ Pixelink camera initialized successfully")
                return True
            else:
                self.pixelink_status.config(text="❌ Init Failed", fg='red')
                if hasattr(self, 'live_indicator'):
                    self.live_indicator.configure(text="❌ Init Failed", fg='red')
                print("⚠ Pixelink camera initialization failed")
                return False
        except Exception as e:
            self.pixelink_status.config(text="🔴 Init Error", fg='red')
            if hasattr(self, 'live_indicator'):
                self.live_indicator.configure(text="🔴 Error", fg='red')
            print(f"✗ Pixelink initialization error: {e}")
            return False

    def start_pixelink(self):
        """Start Pixelink stream with enhanced status updates"""
        try:
            if hasattr(self.spectrometer_manager, 'hCamera') and self.spectrometer_manager.hCamera:
                self.spectrometer_manager.start()
                self.pixelink_status.config(text="🟢 Live Streaming", fg='lightgreen')
                if hasattr(self, 'live_indicator'):
                    self.live_indicator.configure(text="🟢 Live", fg='lightgreen')
                print("✓ Pixelink live stream started")
            else:
                self.pixelink_status.config(text="❌ No Camera", fg='red')
                if hasattr(self, 'live_indicator'):
                    self.live_indicator.configure(text="⚫ No Camera", fg='red')
                print("✗ No Pixelink camera available")
        except Exception as e:
            self.pixelink_status.config(text="🔴 Stream Error", fg='red')
            if hasattr(self, 'live_indicator'):
                self.live_indicator.configure(text="🔴 Error", fg='red')
            print(f"✗ Pixelink stream error: {e}")

    def stop_pixelink(self):
        """Stop Pixelink stream with enhanced status updates"""
        try:
            self.spectrometer_manager.stop()
            self.pixelink_status.config(text="⏹️ Stopped", fg='yellow')
            if hasattr(self, 'live_indicator'):
                self.live_indicator.configure(text="⚫ Offline", fg='red')
            print("✓ Pixelink stream stopped")
        except Exception as e:
            self.pixelink_status.config(text="🔴 Stop Error", fg='red')
            if hasattr(self, 'live_indicator'):
                self.live_indicator.configure(text="🔴 Error", fg='red')
            print(f"✗ Pixelink stop error: {e}")

                
        except Exception as e:
            print(f"Pixelink display update error: {e}")

    def stop_measurement_sequence(self):
        """Stop running measurement sequence"""
        if self._sequence_running:
            self._sequence_stop_requested = True
            print("🛑 STOPPING MEASUREMENT SEQUENCE...")
            # Update button states
            if hasattr(self, 'start_seq_btn'):
                self.start_seq_btn.config(state=NORMAL)
            if hasattr(self, 'stop_seq_btn'):
                self.stop_seq_btn.config(state=DISABLED)

    def start_measurement_sequence(self):
        """Start automated measurement sequence"""
        print("🔥 SEQUENCE BUTTON HAS BEEN CLICKED!")
        
        def sequence():
            try:
                # Set sequence running flags
                self._sequence_running = True
                self._sequence_stop_requested = False
                
                # Update button states
                if hasattr(self, 'start_seq_btn'):
                    self.start_seq_btn.config(state=DISABLED)
                if hasattr(self, 'stop_seq_btn'):
                    self.stop_seq_btn.config(state=NORMAL)
                
                print("🚀 STARTING MEASUREMENT SEQUENCE...")
                print(f"Kalibracja: {getattr(self, 'calibration', False)}")
                
                # Guards: require calibration
                if not getattr(self, 'calibration', False):
                    print("❌ ERROR: Calibration required. Perform calibration in Camera & Controls tab.")
                    return
                
                # Check motor connection
                if not self.motor_controller.connected:
                    print("ERROR: Motors are not connected!")
                    print("Check serial port connections in Settings tab.")
                    return
                    
                # Check camera/spectrometer connection  
                camera_connected = False
                if hasattr(self, 'camera_manager') and self.camera_manager:
                    try:
                        camera_connected = (self.camera_manager.detector is not None and 
                                          self.camera_manager.detector.isOpened())
                    except:
                        camera_connected = False
                
                if not camera_connected and not hasattr(self, 'pixelink_image_data'):
                    print("ERROR: Camera/spectrometer is not connected!")
                    print("Check device connection.")
                    return
                
                # Create data folder
                folder = "measurement_data"
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"measurement_{time.strftime('%Y%m%d_%H%M%S')}_spectra.csv")
                
                # Get image dimensions for scan parameters
                if hasattr(self, 'pixelink_image_data') and self.pixelink_image_data is not None:
                    h, w = self.pixelink_image_data.shape[:2]
                    x, y = 0, 0  # Start from top-left corner
                else:
                    # Default image size if no camera data
                    w, h = 640, 480
                    x, y = 0, 0
                    
                step_x = self.step_x.get()
                step_y = self.step_y.get()
                
                # Calculate number of scan points
                nx = max(1, (w // step_x) + 1)
                ny = max(1, (h // step_y) + 1)
                total_points = nx * ny
                
                print(f"📐 Scan area: {w}x{h} px, starting from ({x}, {y})")
                print(f"📊 Krok skanowania: {step_x}x{step_y} px")
                print(f"🎯 Scan points: {nx} x {ny} = {total_points} points")
                
                # Initialize progress tracking
                current_point = 0
                start_time = time.time()
                
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    count = len(self.spectrum_data)
                    writer.writerow(['measurement_id', 'x_pixel', 'y_pixel', 'x_motor', 'y_motor'] + 
                                  [f'wavelength_{i}' for i in range(count)])
                    
                    print("🏁 Starting square scan from corner...")
                    
                    # Use settings from options.json for scanning area
                    scan_step_x = self.step_x.get()
                    scan_step_y = self.step_y.get()
                    scan_width = self.scan_width.get()
                    scan_height = self.scan_height.get()
                    
                    print(f"📋 Scan parameters from settings:")
                    print(f"   Krok X: {scan_step_x}, Krok Y: {scan_step_y}")
                    print(f"   Width: {scan_width}, Height: {scan_height}")
                    
                    # Calculate number of points in each direction
                    points_x = scan_width // scan_step_x + 1
                    points_y = scan_height // scan_step_y + 1
                    total_points = points_x * points_y
                    
                    print(f"📐 Scan grid: {points_x} x {points_y} = {total_points} points")
                    
                    # Calculate offset to move to top-left corner of scan area
                    # From current position, move to corner where scanning will start
                    offset_x = -scan_width // 2
                    offset_y = -scan_height // 2
                    
                    print(f"📍 Move to corner from current position: ({offset_x}, {offset_y})")
                    
                    # Move to top-left corner of scan area
                    if offset_x != 0:
                        dir_x = 'l' if offset_x < 0 else 'r'
                        self.motor_controller.move(dir_x, abs(offset_x))
                    if offset_y != 0:
                        dir_y = 'd' if offset_y < 0 else 'u'
                        self.motor_controller.move(dir_y, abs(offset_y))
                    
                    time.sleep(1)
                    print("✅ At corner - ready to scan")
                    
                    current_point = 0
                    
                    # Main scanning loop - square grid from corner
                    for iy in range(points_y):
                        # Check for stop request
                        if self._sequence_stop_requested:
                            print("🛑 Sequence stopped by user")
                            break
                            
                        print(f"📍 Scanning row {iy + 1}/{points_y}")
                        
                        for ix in range(points_x):
                            # Check for stop request
                            if self._sequence_stop_requested:
                                print("🛑 Sequence stopped by user")
                                break
                                
                            current_point += 1
                            
                            # Calculate absolute position in grid (starting from 0,0 at corner)
                            grid_x = ix * scan_step_x
                            grid_y = iy * scan_step_y
                            
                            print(f"➡️ Punkt {current_point}: siatka ({grid_x}, {grid_y})")
                            
                            # Move to grid position (only if not the first point)
                            if current_point > 1:
                                # Calculate movement needed from previous point
                                if ix == 0 and iy > 0:  # New row, move down and reset X
                                    self.motor_controller.move('d', scan_step_y)
                                    if (points_x - 1) * scan_step_x > 0:  # Move back to start of row
                                        self.motor_controller.move('l', (points_x - 1) * scan_step_x)
                                elif ix > 0:  # Same row, move right
                                    self.motor_controller.move('r', scan_step_x)
                            
                            # Wait for motor to stabilize
                            time.sleep(0.01)
                            
                            # Acquire spectrum data
                            spectrum = self.spectrum_data.copy()
                            xmin = int(self.xmin_var.get())
                            xmax = int(self.xmax_var.get())
                            spectrum_roi = spectrum[xmin:xmax]
                            
                            # Save measurement data with grid coordinates
                            writer.writerow([current_point, grid_x, grid_y, grid_x, grid_y] + 
                                          spectrum_roi.tolist())
                            
                            # Progress update
                            elapsed = time.time() - start_time
                            progress = (current_point / total_points) * 100
                            eta = (elapsed / current_point * (total_points - current_point)) if current_point > 0 else 0
                            
                            print(f"📊 Punkt {current_point}/{total_points} ({progress:.1f}%) - "
                                  f"Siatka: ({grid_x}, {grid_y}) - ETA: {eta:.0f}s")
                            
                            # Configurable delay for sequence measurements
                            sequence_sleep = options.get('sequence_sleep', 0.1)
                            time.sleep(sequence_sleep)
                
                # Return to original position (before scan started)
                print("🔙 Returning to position before scan...")
                
                # First move back to corner (where we started scanning)
                final_grid_x = (points_x - 1) * scan_step_x
                final_grid_y = (points_y - 1) * scan_step_y
                
                # Move from final position back to corner
                if final_grid_x > 0:
                    self.motor_controller.move('l', final_grid_x)
                if final_grid_y > 0:
                    self.motor_controller.move('u', final_grid_y)
                
                # Then move back to original position (reverse the initial offset)
                if offset_x != 0:
                    dir_x_back = 'r' if offset_x < 0 else 'l'
                    self.motor_controller.move(dir_x_back, abs(offset_x))
                if offset_y != 0:
                    dir_y_back = 'u' if offset_y < 0 else 'd'
                    self.motor_controller.move(dir_y_back, abs(offset_y))
                
                print("✅ Returned to position before scan")
                
                total_time = time.time() - start_time
                print(f"✅ SCAN COMPLETED!")
                print(f"📁 Saved {total_points} measurements to: {filename}")
                print(f"⏱️ Czas skanowania: {total_time:.1f} sekund")
                print(f"⚡ Average {total_time/total_points:.2f} s/point")
                
                # Reload measurements
                self.after(100, self.load_measurements)
                
            except Exception as e:
                print(f"Sequence error: {e}")
            finally:
                # Reset sequence flags and button states
                self._sequence_running = False
                self._sequence_stop_requested = False
                if hasattr(self, 'start_seq_btn'):
                    self.start_seq_btn.config(state=NORMAL)
                if hasattr(self, 'stop_seq_btn'):
                    self.stop_seq_btn.config(state=DISABLED)
        
        # Run sequence in separate thread regardless of motor connection
        if not self.motor_controller.connected:
            print("Motor controller not connected — running in simulation (no moves).")
        threading.Thread(target=sequence, daemon=True).start()

    def apply_settings(self):
        """Apply and save settings"""
        global options  # Move global declaration to the beginning
        
        print("🔧 Applying settings...")
        
        # Debug current values
        print(f"🔍 Current port values: X='{self.port_x_var.get()}', Y='{self.port_y_var.get()}'")
        print(f"🔍 Current step values: X={self.step_x.get()}, Y={self.step_y.get()}")
        
        settings = {
            'step_x': self.step_x.get(),
            'step_y': self.step_y.get(),
            'width': self.scan_width.get(),
            'height': self.scan_height.get(),
            'xmin': self.xmin_var.get(),
            'xmax': self.xmax_var.get(),
            'port_x': self.port_x_var.get(),
            'port_y': self.port_y_var.get(),
            'sequence_sleep': self.sequence_sleep_var.get() if hasattr(self, 'sequence_sleep_var') else options.get('sequence_sleep', 0.1),
            'camera_index': int(self.camera_combo.get()) if hasattr(self, 'camera_combo') and self.camera_combo.get() != '' else options.get('camera_index', 0),
            'cal_px_per_step_x': float(self.cal_x_var.get()) if hasattr(self, 'cal_x_var') else options.get('cal_px_per_step_x', 0.0),
            'cal_px_per_step_y': float(self.cal_y_var.get()) if hasattr(self, 'cal_y_var') else options.get('cal_px_per_step_y', 0.0),
            'cal_sign_x': int(getattr(self, 'cal_sign_x', options.get('cal_sign_x', 1))),
            'cal_sign_y': int(getattr(self, 'cal_sign_y', options.get('cal_sign_y', 1))),
            'lambda_min': float(self.lambda_min_var.get()) if hasattr(self, 'lambda_min_var') else options.get('lambda_min', 400.0),
            'lambda_max': float(self.lambda_max_var.get()) if hasattr(self, 'lambda_max_var') else options.get('lambda_max', 700.0),
            'lambda_calibration_enabled': bool(self.lambda_cal_enabled_var.get()) if hasattr(self, 'lambda_cal_enabled_var') else options.get('lambda_calibration_enabled', True),
            'await': 0.01
        }
        
        try:
            with open('options.json', 'w') as f:
                json.dump(settings, f, indent=4)
            print("✅ Settings saved successfully")
            print(f"📂 Saved ports: X={settings['port_x']}, Y={settings['port_y']}")
            print(f"📂 Saved steps: X={settings['step_x']}, Y={settings['step_y']}")
            print(f"📂 Saved scan area: {settings['width']}x{settings['height']}")
            
            # Update global options (global already declared at function start)
            options.update(settings)
            
            # Reinitialize motor controller with new ports
            self.motor_controller.close()
            self.motor_controller = MotorController(
                self.port_x_var.get(),
                self.port_y_var.get()
            )

            # Switch camera if needed
            new_cam = settings.get('camera_index', 0)
            if new_cam != self.camera_index:
                self.camera_manager.stop()
                self.camera_manager = CameraManager(camera_index=new_cam)
                self.camera_index = new_cam
                print(f"Camera set to index {new_cam}")
            
        except Exception as e:
            print(f"Settings save error: {e}")

    def start_calibration(self, axis):
        # This simple manual calibration will store two clicks before/after a known step
        # User flow: Click center on image, press move in axis, click center again, we compute px/step
        if axis not in ('x', 'y'):
            return
        self._calibration_mode = f"{axis}_before"
        print(f"Kalibracja {axis.upper()} — kliknij punkt referencyjny na obrazie.")
        # Canvas display was removed - calibration disabled
        print("Canvas display removed - manual calibration unavailable")

    def start_calibration_both(self):
        try:
            # Check motor connection first
            if not self.motor_controller.connected:
                print("ERROR: Motors are not connected!")
                print("Check serial port connections in Settings tab.")
                print("X and Y ports must be available for calibration.")
                return
                
            # Check camera connection
            camera_connected = False
            if hasattr(self, 'camera_manager') and self.camera_manager:
                try:
                    camera_connected = (self.camera_manager.detector is not None and 
                                      self.camera_manager.detector.isOpened())
                except:
                    camera_connected = False
            
            if not camera_connected and not hasattr(self, 'pixelink_image_data'):
                print("ERROR: Camera is not connected and no test image available!")
                print("Check camera connection or use test image.")
                return
                
            print("Automatic calibration: performing X and Y movements, analyzing image changes...")
            # Clear previous calibration data
            self._calibration_points = {}
            self._cal_images = {}
            self._last_move_dir = None
            self._calibration_mode = 'auto_running'
            
            # Update button states - disable sequence during calibration
            self._update_start_seq_state()
            
            # Start automatic calibration sequence
            self.after(100, self._auto_calibration_sequence)
        except Exception as e:
            print(f"Auto calibration error: {e}")

    def _auto_calibration_sequence(self):
        """Automatic calibration: move motors, analyze image changes"""
        try:
            if self._calibration_mode == 'auto_running':
                # Step 1: Capture reference image
                print("Step 1: Saving reference image...")
                self._cal_images['reference'] = self.pixelink_image_data.copy()
                
                # Step 2: Move X axis
                print("Step 2: Performing +X movement...")
                step_x = max(1, int(options.get('cal_step_x', 1)))
                self.motor_controller.move('r', step_x)
                self._calibration_mode = 'wait_x'
                # Wait for movement to complete and image to update
                self.after(1000, self._auto_calibration_sequence)
                
            elif self._calibration_mode == 'wait_x':
                # Step 3: Capture image after X movement
                print("Step 3: Analyzing change after X movement...")
                self._cal_images['after_x'] = self.pixelink_image_data.copy()
                
                # Analyze X displacement
                self._analyze_x_displacement()
                
                # Step 4: Move Y axis
                print("Step 4: Performing +Y movement...")
                step_y = max(1, int(options.get('cal_step_y', 1)))
                self.motor_controller.move('u', step_y)
                self._calibration_mode = 'wait_y'
                # Wait for movement to complete
                self.after(1000, self._auto_calibration_sequence)
                
            elif self._calibration_mode == 'wait_y':
                # Step 5: Capture image after Y movement and finalize
                print("Step 5: Analyzing change after Y movement...")
                self._cal_images['after_y'] = self.pixelink_image_data.copy()
                
                # Analyze Y displacement
                self._analyze_y_displacement()
                
                # Finalize calibration
                self._finalize_auto_calibration()
                
        except Exception as e:
            print(f"Auto calibration sequence error: {e}")
            self._calibration_mode = None
    
    def _analyze_x_displacement(self):
        """Analyze image displacement in X direction"""
        try:
            ref_img = self._cal_images['reference']
            after_img = self._cal_images['after_x']
            
            # Use optical flow or correlation to find displacement
            # Simplified: compare image centers
            h, w = ref_img.shape[:2]
            center_h = h // 2
            roi_height = min(50, h // 4)
            
            ref_strip = ref_img[center_h-roi_height:center_h+roi_height, :]
            after_strip = after_img[center_h-roi_height:center_h+roi_height, :]
            
            # Cross-correlation to find shift
            correlation = cv2.matchTemplate(after_strip.astype(np.float32), 
                                          ref_strip.astype(np.float32), 
                                          cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Calculate displacement in pixels
            displacement_x = max_loc[0] - (w // 2)
            steps = max(1, int(options.get('cal_step_x', 1)))
            
            if abs(displacement_x) > 0.1:  # Minimum detectable movement
                self.cal_px_per_step_x = abs(displacement_x) / steps
                self.cal_sign_x = 1 if displacement_x > 0 else -1
                print(f"X calibration: {displacement_x:.1f}px displacement, {self.cal_px_per_step_x:.3f} px/step")
            else:
                print("X: No image displacement detected")
                
        except Exception as e:
            print(f"X analysis error: {e}")
    
    def _analyze_y_displacement(self):
        """Analyze image displacement in Y direction"""
        try:
            ref_img = self._cal_images['after_x']  # Use after X as reference for Y
            after_img = self._cal_images['after_y']
            
            h, w = ref_img.shape[:2]
            center_w = w // 2
            roi_width = min(50, w // 4)
            
            ref_strip = ref_img[:, center_w-roi_width:center_w+roi_width]
            after_strip = after_img[:, center_w-roi_width:center_w+roi_width]
            
            # Cross-correlation to find shift
            correlation = cv2.matchTemplate(after_strip.astype(np.float32), 
                                          ref_strip.astype(np.float32), 
                                          cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Calculate displacement in pixels
            displacement_y = max_loc[1] - (h // 2)
            steps = max(1, int(options.get('cal_step_y', 1)))
            
            if abs(displacement_y) > 0.1:  # Minimum detectable movement
                self.cal_px_per_step_y = abs(displacement_y) / steps
                self.cal_sign_y = 1 if displacement_y > 0 else -1
                print(f"Y calibration: {displacement_y:.1f}px displacement, {self.cal_px_per_step_y:.3f} px/step")
            else:
                print("Y: No image displacement detected")
                
        except Exception as e:
            print(f"Y analysis error: {e}")
    
    def _finalize_auto_calibration(self):
        """Complete automatic calibration"""
        try:
            # Update calibration displays
            if hasattr(self, 'cal_x_var'):
                self.cal_x_var.set(self.cal_px_per_step_x)
            if hasattr(self, 'cal_y_var'):
                self.cal_y_var.set(self.cal_px_per_step_y)
            
            # Set calibration complete
            self.calibration = bool(self.cal_px_per_step_x > 0 and self.cal_px_per_step_y > 0)
            
            if self.calibration:
                print("✓ Automatic calibration completed successfully!")
                print(f"X: {self.cal_px_per_step_x:.3f} px/step, sign: {self.cal_sign_x}")
                print(f"Y: {self.cal_px_per_step_y:.3f} px/step, sign: {self.cal_sign_y}")
                print("✓ Ready to start measurement sequence")
                
                # Auto-save calibration
                self.save_calibration()
                
                # Generate and store scan points for visualization
                self.scan_points = self.generate_scan_points()
                print(f"✓ Generated {len(self.scan_points)} scan points")
                print("✓ Scanning will be performed on the entire image surface")
                
            else:
                print("ERROR: Calibration failed - no image movement detected")
            
            # Update UI states
            self._update_start_seq_state()
            self._calibration_mode = None
            
        except Exception as e:
            print(f"Finalization error: {e}")
            self._calibration_mode = None
    
    def generate_scan_points(self):
        """Generate scan points based on full image and step settings"""
        # Use full image dimensions
        if hasattr(self, 'pixelink_image_data') and self.pixelink_image_data is not None:
            h, w = self.pixelink_image_data.shape[:2]
            x, y = 0, 0  # Start from top-left corner
        else:
            # Default image size if no camera data
            w, h = 640, 480
            x, y = 0, 0
            
        step_x = self.step_x.get()
        step_y = self.step_y.get()
        
        # Calculate number of scan points
        nx = max(1, (w // step_x) + 1)
        ny = max(1, (h // step_y) + 1)
        
        points = []
        
        # Generate points using the same raster pattern as scanning sequence
        for iy in range(ny):
            for ix in range(nx):
                # Calculate pixel positions (same formula as scanning)
                pixel_x = x + (ix * step_x)
                pixel_y = y + (iy * step_y)
                points.append((pixel_x, pixel_y))
        
        print(f"🎯 Generated {len(points)} scan points ({nx}x{ny} grid)")
        return points
        
    def reset_calibration(self):
        self.cal_px_per_step_x = 0.0
        self.cal_px_per_step_y = 0.0
        if hasattr(self, 'cal_x_var'):
            self.cal_x_var.set(0.0)
        if hasattr(self, 'cal_y_var'):
            self.cal_y_var.set(0.0)
        # Remove calibration markers
        try:
            for k, (pt, lbl) in list(self._cal_markers.items()):
                try:
                    pt.remove()
                except Exception:
                    pass
                try:
                    lbl.remove()
                except Exception:
                    pass
                self._cal_markers.pop(k, None)
            # Canvas display was removed - skip drawing
            pass
        except Exception:
            pass
        print("Reset kalibracji.")

    def _add_cal_marker(self, key, x, y, color, label):
        # Remove existing marker for this key
        try:
            if key in self._cal_markers:
                old_pt, old_lbl = self._cal_markers[key]
                try:
                    old_pt.remove()
                except Exception:
                    pass
                try:
                    old_lbl.remove()
                except Exception:
                    pass
        except Exception:
            pass
        # Add new marker and label
        # Canvas display was removed - skip marker display
        print(f"Calibration marker {label} at ({x}, {y}) - display disabled")
        try:
            pass  # Canvas drawing disabled
        except Exception:
            pass
    
    def load_measurements(self):
        """Load measurement files list without caching data"""
        folder = "measurement_data"
        self.measurement_files = []  # Store only filenames, not data
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Just collect filenames - don't load data into memory
        for filename in sorted(glob.glob(os.path.join(folder, "*_spectra.csv"))):
            self.measurement_files.append(filename)
            
        self.draw_measurements()
    
    def _load_measurement_data_on_demand(self, filename):
        """Load measurement data only when needed"""
        data = []
        try:
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        x = int(row[0])
                        y = int(row[1])
                        spectrum = [float(v) for v in row[2:]]
                        data.append([x, y, spectrum])
                    except Exception as e:
                        print(f"Skipped row with error: {row} ({e})")
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
        return data
    
    def export_measurements(self):
        """Export all measurements to a single file"""
        if not self.measurement_files:
            messagebox.showinfo("Info", "No measurements to export")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Export Measurements",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['measurement_id', 'x', 'y'] + [f'wavelength_{i}' for i in range(2048)])
                    
                    # Write all measurements - load on demand
                    for measurement_id, measurement_file in enumerate(self.measurement_files):
                        measurement_data = self._load_measurement_data_on_demand(measurement_file)
                        for point in measurement_data:
                            x, y, spectrum = point
                            writer.writerow([measurement_id + 1, x, y] + spectrum)
                    
                    print(f"Exported {len(self.measurement_files)} measurements to {filename}")
                    messagebox.showinfo("Success", f"Measurements exported to:\n{filename}")
                    
            except Exception as e:
                print(f"Export error: {e}")
                messagebox.showerror("Error", f"Cannot export measurements:\n{e}")

    def delete_all_measurements(self):
        """Delete all measurements"""
        if not self.measurement_files:
            messagebox.showinfo("Info", "No measurements to delete")
            return
        
        result = messagebox.askyesno(
            "Delete All Measurements",
            f"Are you sure you want to delete all {len(self.measurement_files)} measurements?\n"
            "This action cannot be undone!"
        )
        
        if result:
            try:
                folder = "measurement_data"
                deleted_count = 0
                
                # Delete all CSV files
                for filename in glob.glob(os.path.join(folder, "*_spectra.csv")):
                    if os.path.exists(filename):
                        os.remove(filename)
                        deleted_count += 1
                
                self.measurement_files.clear()  # Clear file list, not data cache
                self.draw_measurements()
                
                print(f"Deleted {deleted_count} measurement files")
                messagebox.showinfo("Success", f"Deleted {deleted_count} measurements")
                
            except Exception as e:
                print(f"Error deleting measurements: {e}")
                messagebox.showerror("Error", f"Cannot delete measurements:\n{e}")

    def delete_measurement(self, measurement_index):
        """Delete selected measurement"""
        if 0 <= measurement_index < len(self.measurement_files):
            result = messagebox.askyesno(
                "Delete Measurement",
                f"Are you sure you want to delete measurement {measurement_index + 1}?\n"
                "This action cannot be undone!"
            )
            
            if result:
                try:
                    # Delete the specific file
                    file_to_delete = self.measurement_files[measurement_index]
                    os.remove(file_to_delete)
                    print(f"Deleted file: {os.path.basename(file_to_delete)}")
                    
                    # Remove from file list and refresh
                    self.measurement_files.pop(measurement_index)
                    self.draw_measurements()
                    
                except Exception as e:
                    print(f"Error deleting measurement: {e}")
                    messagebox.showerror("Error", f"Cannot delete measurement:\n{e}")

    def draw_measurements(self):
        """Draw measurement buttons in grid layout"""
        # Clear existing buttons
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.measurement_files:
            # Show message if no measurements
            Label(
                self.results_frame, 
                text="No measurements\nRun measurement sequence to create data",
                bg=self.DGRAY, fg='lightgray', font=("Arial", 12),
                justify=CENTER
            ).grid(row=0, column=0, padx=20, pady=20)
        else:
            # Create grid of measurement buttons
            buttons_per_row = 5  # Number of buttons per row
            
            for i, filename in enumerate(self.measurement_files):
                row = i // buttons_per_row
                col = i % buttons_per_row
                
                # Create button frame for better styling
                button_frame = Frame(self.results_frame, bg=self.DGRAY, relief='raised', bd=1)
                button_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                
                # Main button
                btn = CButton(
                    button_frame,
                    text=f"Pomiar {i+1}",
                    command=lambda idx=i: self.show_measurement_by_index(idx),
                    width=12, height=2,
                    font=("Arial", 10, "bold")
                )
                btn.pack(fill=BOTH, expand=True, padx=2, pady=2)
                
                # Info label with filename
                basename = os.path.basename(filename)
                info_label = Label(
                    button_frame,
                    text=basename.replace('_spectra.csv', ''),
                    bg=self.RGRAY, fg='lightgray',
                    font=("Arial", 8), justify=CENTER
                )
                info_label.pack(fill=X, padx=2, pady=(0, 2))
                
                # Delete button (small)
                delete_btn = Button(
                    button_frame,
                    text="×",
                    command=lambda idx=i: self.delete_measurement(idx),
                    bg='darkred', fg='white', font=("Arial", 8, "bold"),
                    width=2, height=1, bd=0
                )
                delete_btn.pack(side=RIGHT, anchor='ne', padx=2, pady=2)
            
            # Configure grid weights for proper resizing
            for i in range(buttons_per_row):
                self.results_frame.columnconfigure(i, weight=1)
        
        # Update info label
        if hasattr(self, 'results_info'):
            self.results_info.config(text=f"Pomiary: {len(self.measurement_files)}")

    def show_measurement_by_index(self, measurement_index):
        """Show selected measurement by index - load data on demand"""
        if 0 <= measurement_index < len(self.measurement_files):
            filename = self.measurement_files[measurement_index]
            # Load data only when needed
            measurement_data = self._load_measurement_data_on_demand(filename)
            HeatMapWindow(self, measurement_index + 1, measurement_data, self.current_image)

    def move_motor(self, direction):
        """Manual motor movement function"""
        try:
            step_size = self.motor_step_var.get()
            
            if not self.motor_controller.connected:
                self.motor_status.config(text=f"Motor Status: Not connected - simulated move {direction} {step_size}")
                print(f"🔧 Manual move: {direction} {step_size} steps (simulation)")
                return
            
            # Update status
            self.motor_status.config(text=f"Motor Status: Moving {direction} {step_size} steps...")
            print(f"🔧 Manual move: {direction} {step_size} steps")
            
            # Execute movement
            if direction == 'o':
                self.motor_controller.move('o')  # Origin
            else:
                self.motor_controller.move(direction, step_size)
            
            # Update status after movement
            self.after(200, lambda: self.motor_status.config(text="Motor Status: Ready"))
            
        except Exception as e:
            error_msg = f"Motor Status: Error - {e}"
            self.motor_status.config(text=error_msg)
            print(f"❌ Motor movement error: {e}")

    # Duplicate cleanup removed; using unified cleanup above


if __name__ == "__main__":
    try:
        print("=== URUCHAMIANIE APLIKACJI ===")
        app = SpektrometerApp()
        print("✅ Aplikacja utworzona")
        
        print("🚀 Starting main loop...")
        app.mainloop()
        print("✅ Main loop completed")
        
    except Exception as e:
        print(f"❌ APPLICATION ERROR: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("⚠️ Application interrupted by user")
    finally:
        try:
            if 'app' in locals():
                print("🧹 Cleaning up resources...")
                app.cleanup()
                print("✅ Zasoby wyczyszczone")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
