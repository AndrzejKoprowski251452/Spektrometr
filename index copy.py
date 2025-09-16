"""
Spektrometr Application - Main File
Refactored with proper threading and structure
"""

import os
import sys
import time
import threading
import queue
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
import serial.tools.list_ports

# Local imports
from pixelinkWrapper import *
from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
from async_tkinter_loop import async_mainloop

# Load configuration
try:
    with open('options.json', 'r') as f:
        options = json.load(f)
except FileNotFoundError:
    options = {
        'step_x': 10, 'step_y': 10, 'offset': 5,
        'width': 100, 'height': 100, 'await': 0.01,
        'xmin': '0', 'xmax': '2048',
        'port_x': 'COM5', 'port_y': 'COM9',
        'port_aux1': '', 'port_aux2': '',
        'camera_index': 0,
        'cal_px_per_step_x': 0.0, 'cal_px_per_step_y': 0.0,
        'cal_sign_x': 1, 'cal_sign_y': 1,
        'cal_step_x': 1, 'cal_step_y': 1
    }

# Color constants
LGRAY = '#232323'
DGRAY = '#161616'
RGRAY = '#2c2c2c'
MGRAY = '#1D1c1c'


class ThreadSafeQueue:
    """Thread-safe queue for communication between threads"""
    def __init__(self):
        self.queue = queue.Queue()
    
    def put(self, item):
        self.queue.put(item)
    
    def get(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


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
        self.frame_queue = ThreadSafeQueue()
        self.direction_queue = ThreadSafeQueue()
        self.thread = None
        
    def start(self):
        """Start camera thread"""
        if not self.running:
            self.running = True
            self.detector = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.thread.start()
    
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
                
                # Put frame and direction in queues
                self.frame_queue.put(frame)
                self.direction_queue.put(direction)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)


class SpectrometerManager:
    """Manages spectrometer operations in separate thread"""
    
    def __init__(self):
        self.hCamera = None
        self.running = False
        self.thread = None
        self.data_queue = ThreadSafeQueue()
        
    def initialize(self):
        """Initialize spectrometer camera - based on samples pattern"""
        try:
            print("Attempting to initialize Pixelink camera...")
            
            # Simple initialization like in samples
            ret = PxLApi.initialize(0)
            print(f"PxLApi.initialize(0) returned: {ret[0]}")
            
            if PxLApi.apiSuccess(ret[0]):
                self.hCamera = ret[1]
                print(f"Camera handle obtained: {self.hCamera}")
                
                # CRITICAL: Reset to factory settings first - required for proper operation
                print("Resetting camera to factory settings...")
                ret_factory = PxLApi.loadSettings(self.hCamera, PxLApi.Settings.SETTINGS_FACTORY)
                if PxLApi.apiSuccess(ret_factory[0]):
                    print("Factory settings loaded successfully")
                else:
                    print(f"Warning: Factory settings load failed: {ret_factory[0]}")
                
                print("Pixelink camera initialized successfully")
                return True
            else:
                print(f"Failed to initialize camera: {ret[0]} - No Pixelink camera detected")
                return False
                
        except Exception as e:
            print(f"Spectrometer initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self):
        """Start spectrometer data collection"""
        if self.hCamera and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._spectrometer_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop spectrometer - based on samples pattern"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.hCamera:
            try:
                # Stop streaming and uninitialize like in samples
                PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
                PxLApi.uninitialize(self.hCamera)
                print("Pixelink camera stopped and uninitialized")
            except Exception as e:
                print(f"Error stopping camera: {e}")
    
    def _spectrometer_loop(self):
        """Spectrometer data collection loop - using getNextNumPyFrame like samples"""
        # Create smaller numpy buffer for frame data - reduce memory usage
        MAX_WIDTH = 2048   # Reduced from 5000
        MAX_HEIGHT = 1536  # Reduced from 5000
        MAX_BYTES_PER_PIXEL = 1  # Grayscale only
        frame = np.zeros([MAX_HEIGHT, MAX_WIDTH*MAX_BYTES_PER_PIXEL], dtype=np.uint8)
        
        print(f"Allocated frame buffer: {frame.shape} = {frame.nbytes / 1024 / 1024:.1f} MB")
        
        # Start streaming like in samples
        ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)
        if not PxLApi.apiSuccess(ret[0]):
            print(f"Failed to start streaming: {ret[0]}")
            return
        
        print("Pixelink streaming started")
        
        while self.running:
            try:
                # Get frame using robust wrapper like in samples
                ret = self._get_next_numpy_frame(self.hCamera, frame, 5)
                if PxLApi.apiSuccess(ret[0]):                    
                    # Get frame descriptor with proper dimensions like in samples
                    frameDescriptor = ret[1]
                    
                    # Put frame data in queue for processing - make smaller copy
                    # Use frame.shape instead of frameDescriptor attributes
                    frame_height, frame_width = frame.shape[:2]
                    frame_copy = frame[:frame_height, :frame_width].copy()
                    self.data_queue.put({
                        'frame': frame_copy,  # Smaller copy of actual frame size
                        'descriptor': frameDescriptor,
                        'timestamp': time.time()
                    })
                else:
                    # Handle errors like in samples
                    if ret[0] == PxLApi.ReturnCode.ApiStreamStopped or \
                       ret[0] == PxLApi.ReturnCode.ApiNoCameraAvailableError:
                        print("Camera stream stopped or camera unavailable")
                        break
                    elif ret[0] == PxLApi.ReturnCode.ApiCameraTimeoutError:
                        pass  # Just retry on timeout
                    else:
                        print(f"getNextNumPyFrame failed: {ret[0]}")
                    
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Spectrometer frame capture error: {e}")
                time.sleep(0.1)
        
        # Stop streaming when done
        PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)

    def _get_next_numpy_frame(self, hCamera, frame, maxNumberOfTries):
        """Robust wrapper around PxLApi.getNextNumPyFrame like in samples"""
        ret = (PxLApi.ReturnCode.ApiUnknownError,)

        for i in range(maxNumberOfTries):
            ret = PxLApi.getNextNumPyFrame(hCamera, frame)
            if PxLApi.apiSuccess(ret[0]):
                return ret
            else:
                # If the streaming is turned off, or worse yet -- is gone?
                # If so, no sense in continuing.
                if PxLApi.ReturnCode.ApiStreamStopped == ret[0] or \
                    PxLApi.ReturnCode.ApiNoCameraAvailableError == ret[0]:
                    return ret

        # Ran out of tries, so return whatever the last error was.
        return ret


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
        self.set_title(f'Pomiar {measurement_index}')
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
        self.xmin = float(self.parent.xmin_var.get())
        self.xmax = float(self.parent.xmax_var.get())
        self.lambdas = np.linspace(self.xmin, self.xmax, spectrum_len)
    
    def _create_widgets(self):
        """Create GUI widgets"""
        self.slider = Scale(
            self.window, from_=0, to=self.cube.shape[2] - 1,
            orient=HORIZONTAL, command=self.on_slider,
            bg=self.DGRAY, fg='lightgray', length=600
        )
        self.slider.pack(fill=X, padx=10, pady=10)
        
        self.fig = plt.figure(figsize=(12, 8), facecolor=self.DGRAY)
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.ax3d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax2d = self.fig.add_subplot(gs[1])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    def on_slider(self, val):
        """Handle slider change"""
        self.current_lambda = int(val)
        self._update_plots()
    
    def _update_plots(self):
        """Update both 3D and 2D plots"""
        # 3D heatmap
        self.ax3d.clear()
        data = self.cube[:, :, self.current_lambda]
        X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
        self.ax3d.plot_surface(X, Y, data.T, cmap='hot')
        
        lambda_val = self.lambdas[self.current_lambda]
        self.ax3d.set_title(f"3D Heatmap - λ={lambda_val:.0f}", color='white')
        self.ax3d.set_xlabel("X", color='white')
        self.ax3d.set_ylabel("Y", color='white')
        self.ax3d.set_zlabel("Intensity", color='white')
        
        # 2D spectrum
        self.ax2d.clear()
        mean_profile = self.cube.mean(axis=(0, 1))
        self.ax2d.plot(self.lambdas, mean_profile, color='orange', label="Average Spectrum")
        self.ax2d.axvline(lambda_val, color='red', linestyle='--', label=f"λ={lambda_val:.0f}")
        self.ax2d.set_title("Spectrum Profile", color='white')
        self.ax2d.set_xlabel("Wavelength", color='white')
        self.ax2d.set_ylabel("Intensity", color='white')
        self.ax2d.legend()
        
        # Style plots
        for ax in [self.ax3d, self.ax2d]:
            ax.set_facecolor(self.DGRAY)
            ax.tick_params(colors='white')
        
        self.fig.tight_layout()
        self.canvas.draw()


class SpektrometerApp(CustomTk):
    """Main application class"""
    
    def __init__(self):
        super().__init__()
        self.title("Spektrometr")
        self.geometry('1400x900')
        
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
        self.roi_active = False
        self.roi_enabled = False
        self.roi_coordinates = None  # New format: (x1, y1, x2, y2)
        self.grid_lines = []
        self.show_grid = False
        self.scan_points = []
        self.show_scan_points = False
        self._scan_point_markers = []
        self._pixelink_event_cids = []
        self._calibration_mode = None  # 'x_before', 'x_after', 'y_before', 'y_after'
        self._calibration_points = {}
        self._cal_markers = {}
        self._cal_images = {}  # Store before/after images for auto-calibration
        self.aux_serial = {}
        self.cal_sign_x = int(options.get('cal_sign_x', 1))
        self.cal_sign_y = int(options.get('cal_sign_y', 1))
        self._last_move_dir = None
        # Calibration state: true only if px/step exist for both axes
        self.calibration = bool(self.cal_px_per_step_x > 0 and self.cal_px_per_step_y > 0)
        
        # Threading for performance - non-blocking operations
        import threading
        self._spectrum_needs_update = False
        self._stop_threads = False
        self._data_lock = threading.Lock()
        
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
        
        # Create tabs - POŁĄCZONE ZAKŁADKI
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
        Label(main_container, text="Live Camera", font=("Arial", 16, "bold"), 
              bg=self.DGRAY, fg='white').pack(pady=(0, 10))
        
        # Centered camera display (Canvas with drag select) - moderate size
        camera_container = Frame(main_container, bg=self.DGRAY)
        camera_container.pack(expand=False, fill=X)
        
        self.camera_canvas = Canvas(camera_container, bg=self.DGRAY, highlightthickness=0)
        self.camera_canvas.pack()
        self._camera_canvas_img = None
        self._camera_frame_size = (800, 600)  # Moderate camera view to leave space for controls
        self.camera_canvas.config(width=self._camera_frame_size[0], height=self._camera_frame_size[1])
        
        # Draw placeholder if no camera
        self._draw_camera_placeholder()
        
        # Drag selection state
        self.cam_drag_start = None
        self.cam_drag_rect = None
        self.cam_scan_area_norm = None  # (x0,y0,x1,y1) normalized 0..1
        self.camera_canvas.bind('<ButtonPress-1>', self._cam_on_press)
        self.camera_canvas.bind('<B1-Motion>', self._cam_on_drag)
        self.camera_canvas.bind('<ButtonRelease-1>', self._cam_on_release)
        
        # Status under canvas
        self.cam_status = Label(main_container, bg=self.DGRAY, fg='lightgray', 
                               text="Camera Status: Not started")
        self.cam_status.pack(pady=5)
        
        # Control buttons frame - centered
        control_frame = Frame(main_container, bg=self.DGRAY)
        control_frame.pack(pady=10)
        
        # Essential controls only
        CButton(control_frame, text="Start Camera", command=self.start_camera).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Stop Camera", command=self.stop_camera).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Calibrate", command=self.start_calibration_both).pack(side=LEFT, padx=10)
        
        # Add Start Sequence button (needed for state management)
        self.start_seq_btn = CButton(control_frame, text="Start Sequence", command=self.start_measurement_sequence)
        self.start_seq_btn.pack(side=LEFT, padx=10)
        # Initial state based on calibration and ROI
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

    def _setup_spectrum_pixelink_tab(self):
        """Setup combined spectrum and pixelink tab"""
        # Create main container
        main_container = Frame(self.tab_spectrum_pixelink, bg=self.DGRAY)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Spectrum (30% of space)
        spectrum_frame = Frame(main_container, bg=self.DGRAY)
        spectrum_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        spectrum_frame.pack_propagate(False)
        spectrum_frame.configure(height=250)  # Fixed smaller height
        
        # Spectrum title and controls
        spectrum_header = Frame(spectrum_frame, bg=self.DGRAY)
        spectrum_header.pack(fill=X, pady=(0, 5))
        
        Label(spectrum_header, text="Live Spectrum", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack(side=LEFT)
        
        # Range controls
        range_frame = Frame(spectrum_header, bg=self.DGRAY)
        range_frame.pack(side=RIGHT)
        
        Label(range_frame, text="X min:", bg=self.DGRAY, fg='white').pack(side=LEFT)
        self.xmin_var = StringVar(value=options.get('xmin', '0'))
        Entry(range_frame, textvariable=self.xmin_var, width=8, bg=self.RGRAY, fg='white').pack(side=LEFT, padx=5)
        
        Label(range_frame, text="X max:", bg=self.DGRAY, fg='white').pack(side=LEFT)
        self.xmax_var = StringVar(value=options.get('xmax', '2048'))
        Entry(range_frame, textvariable=self.xmax_var, width=8, bg=self.RGRAY, fg='white').pack(side=LEFT, padx=5)
        
        CButton(range_frame, text="Set Range", command=self.set_spectrum_range).pack(side=LEFT, padx=5)
        CButton(range_frame, text="Reset View", command=self.reset_spectrum_view).pack(side=LEFT, padx=5)
        CButton(range_frame, text="Auto Scale", command=self.auto_scale_spectrum).pack(side=LEFT, padx=5)
        
        # Spectrum plot
        self._create_spectrum_plot(spectrum_frame)
        
        # Bottom section - Pixelink (70% of space - remaining space)
        pixelink_frame = Frame(main_container, bg=self.DGRAY)
        pixelink_frame.pack(fill=BOTH, expand=True, pady=(5, 0))
        
        # Pixelink title and controls
        pixelink_header = Frame(pixelink_frame, bg=self.DGRAY)
        pixelink_header.pack(fill=X, pady=(0, 5))
        
        Label(pixelink_header, text="Spectrometer Camera View", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack(side=LEFT)
        
        # Pixelink control buttons
        pixelink_controls = Frame(pixelink_header, bg=self.DGRAY)
        pixelink_controls.pack(side=RIGHT)
        
        CButton(pixelink_controls, text="Initialize", command=self.init_pixelink).pack(side=LEFT, padx=2)
        CButton(pixelink_controls, text="Start", command=self.start_pixelink).pack(side=LEFT, padx=2)
        CButton(pixelink_controls, text="Stop", command=self.stop_pixelink).pack(side=LEFT, padx=2)
        CButton(pixelink_controls, text="Reset View", command=self.reset_pixelink_view).pack(side=LEFT, padx=2)
        CButton(pixelink_controls, text="Load Test Image", command=self.load_test_image).pack(side=LEFT, padx=2)
        
        # Status label
        self.pixelink_status = Label(
            pixelink_controls,
            text="Status: Not initialized",
            bg=self.DGRAY, fg='lightgray'
        )
        self.pixelink_status.pack(side=LEFT, padx=10)
        
        # Pixelink plot
        self._create_pixelink_plot(pixelink_frame)

    def _create_spectrum_plot(self, parent_frame):
        """Create spectrum matplotlib plot with navigation toolbar"""
        self.spectrum_fig, self.spectrum_ax = plt.subplots(figsize=(14, 6), facecolor=self.DGRAY)
        self.spectrum_ax.set_facecolor(self.DGRAY)
        
        self.x_axis = np.linspace(0, 2048, 2048)
        self.spectrum_line, = self.spectrum_ax.plot(self.x_axis, self.spectrum_data, color='green', linewidth=1.5)
        
        # Style the plot - same as pixelink
        self.spectrum_ax.set_xlabel("Pixel/Wavelength", color='white', fontsize=12)
        self.spectrum_ax.set_ylabel("Intensity", color='white', fontsize=12)
        self.spectrum_ax.set_title("Live Spectrum", color='white', fontsize=14)
        self.spectrum_ax.tick_params(colors='white')
        self.spectrum_ax.grid(True, alpha=0.3, color='gray')
        
        # Set better margins - same as pixelink
        self.spectrum_fig.tight_layout(pad=2.0)
        
        # Create canvas
        canvas_frame = Frame(parent_frame, bg=self.DGRAY)
        canvas_frame.pack(fill=BOTH, expand=True, pady=5)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=canvas_frame)
        self.spectrum_canvas.draw()
        
        # Navigation toolbar for zoom/pan
        toolbar_frame = Frame(parent_frame, bg=self.DGRAY)
        toolbar_frame.pack(fill=X, pady=2)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.spectrum_toolbar = NavigationToolbar2Tk(self.spectrum_canvas, toolbar_frame)
        self.spectrum_toolbar.config(bg=self.DGRAY)
        
        # Style toolbar
        self._style_toolbar(self.spectrum_toolbar)
        self.spectrum_toolbar.update()
        
        # Pack canvas - centered
        self.spectrum_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Add mouse interaction
        self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
        self.spectrum_canvas.mpl_connect('button_press_event', self.on_spectrum_click)
        
        # Initialize crosshair
        self.spectrum_crosshair_v = None
        self.spectrum_crosshair_h = None
        self.spectrum_annotation = None

    def _create_pixelink_plot(self, parent_frame):
        """Create simplified spectrum area without image display"""
        # Header with title
        pixelink_header = Frame(parent_frame, bg=self.DGRAY)
        pixelink_header.pack(fill=X, pady=(0, 5))
        Label(pixelink_header, text="Pixelink Spectrum Calculator", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack()
        
        # Status display instead of image
        status_frame = Frame(parent_frame, bg=self.DGRAY, relief='sunken', bd=2)
        status_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.pixelink_status = Label(status_frame, bg=self.DGRAY, fg='lightgreen', 
                                   text="Spectrum calculation active\nNo image display", 
                                   font=("Arial", 12), justify=CENTER)
        self.pixelink_status.pack(expand=True, fill=BOTH, padx=20, pady=20)
        
        # Load test image data for spectrum calculation only
        self.pixelink_image_data = self._load_test_image_data()
        self.is_live_streaming = False
        
        # Initialize spectrum with test image data
        if self.pixelink_image_data is not None:
            x, y, w, h = self.get_roi_bounds()
            roi = self.pixelink_image_data[y:y+h, x:x+w]
            if roi.size > 0:
                roi_mean = np.mean(roi, axis=0)
                self.spectrum_data = self._resample_to_2048(roi_mean)
                # Force initial spectrum display
                self.after_idle(self._update_spectrum_plot)

        # Simplified controls - just ROI
        controls = Frame(parent_frame, bg=self.DGRAY)
        controls.pack(fill=X, pady=4)
        self.roi_btn = CButton(controls, text="Select ROI", command=self.toggle_roi_mode)
        self.roi_btn.pack(side=LEFT, padx=2)
        # Initialize ROI button state
        self._update_roi_button_state()
        
        Label(controls, text="ROI: Click to select spectrum area", 
              bg=self.DGRAY, fg='lightgray', font=("Arial", 9)).pack(side=LEFT, padx=10)

        # Remove image event binding
        pass
    
    def _update_pixelink_image_display(self):
        """Disabled - no image display needed"""
        pass
    
    def _finish_image_update(self):
        """Disabled - no image updates"""
        pass

    def _bind_roi_events(self):
        """Disabled - no image interaction needed"""
        pass

    def toggle_roi_mode(self):
        """Toggle ROI selection mode"""
        if not getattr(self, 'calibration', False):
            print("❌ Najpierw przeprowadź kalibrację w zakładce 'Camera & Controls'")
            return
            
        if not getattr(self, 'roi_enabled', False):
            print("❌ ROI selection jest wyłączony. Przeprowadź kalibrację aby go włączyć.")
            return
            
        self.roi_active = not self.roi_active
        
        if self.roi_active:
            print("✓ ROI MODE: WŁĄCZONY - Przeciągnij myszą po obrazie kamery aby zaznaczyć obszar skanowania")
            self.roi_btn.configure(text="✓ ROI Active", bg='#28a745', fg='white')
        else:
            print("ROI MODE: WYŁĄCZONY")
            self.roi_btn.configure(text="Select ROI", bg='#6c757d', fg='white')

    def toggle_grid(self):
        """Toggle grid display - disabled for Tkinter image"""
        self.show_grid = not self.show_grid
        print(f"Grid: {'ON' if self.show_grid else 'OFF'} (visual grid disabled for performance)")
        # Grid drawing disabled for Tkinter Label - saves resources

    def _on_pixelink_press(self, event):
        """Disabled - no image interaction"""
        pass

    def _on_pixelink_drag(self, event):
        """Disabled - no image interaction"""
        pass

    def _on_pixelink_release(self, event):
        """Disabled - no image interaction"""
        pass
        print(f"ROI set: {self.get_roi_bounds()}")
        
        # Regenerate scan points when ROI changes
        if self.calibration:
            self.scan_points = self.generate_scan_points()
            print(f"Wygenerowano {len(self.scan_points)} punktów skanowania dla nowego ROI")
        
        self._update_start_seq_state()

    def _update_start_seq_state(self):
        try:
            # Check conditions
            cal_status = getattr(self, 'calibration', False)
            roi_status = hasattr(self, 'roi_coordinates') and self.roi_coordinates is not None
            
            enabled = bool(cal_status and roi_status)
            
            if hasattr(self, 'start_seq_btn'):
                old_state = str(self.start_seq_btn.cget('state'))
                self.start_seq_btn.configure(state=NORMAL if enabled else DISABLED)
                
                # Update button appearance and text based on state
                if enabled:
                    self.start_seq_btn.configure(
                        text="▶ START SEQUENCE", 
                        bg='#28a745',  # Green background when enabled
                        fg='white'
                    )
                    # Print message when sequence button becomes enabled
                    if old_state == 'disabled':
                        print("✅ PRZYCISK SEKWENCJI ODBLOKOWANY - Można rozpocząć pomiary!")
                        print("Kliknij '▶ START SEQUENCE' aby rozpocząć automatyczne skanowanie")
                else:
                    self.start_seq_btn.configure(
                        text="⏸ Start Sequence", 
                        bg='#6c757d',  # Gray background when disabled
                        fg='lightgray'
                    )
        except Exception as e:
            print(f"Błąd aktualizacji stanu przycisku: {e}")

    def _update_roi_button_state(self):
        """Update ROI button state based on calibration"""
        try:
            if hasattr(self, 'roi_btn'):
                calibrated = getattr(self, 'calibration', False)
                roi_enabled = getattr(self, 'roi_enabled', False)
                
                if calibrated and roi_enabled:
                    self.roi_btn.configure(state=NORMAL, bg='#007bff', fg='white')
                    print("✓ Przycisk ROI odblokowany - można zaznaczać obszar skanowania")
                else:
                    self.roi_btn.configure(state=DISABLED, bg='#6c757d', fg='lightgray')
        except Exception as e:
            print(f"Błąd aktualizacji przycisku ROI: {e}")

    def _draw_grid(self):
        """Grid drawing disabled for Tkinter Label - no visual feedback needed"""
        pass

    def get_roi_bounds(self):
        """Get ROI bounds - uses roi_coordinates if available"""
        if hasattr(self, 'roi_coordinates') and self.roi_coordinates:
            x1, y1, x2, y2 = self.roi_coordinates
            return (x1, y1, x2-x1, y2-y1)
        elif hasattr(self, 'pixelink_image_data') and self.pixelink_image_data is not None:
            return (0, 0, self.pixelink_image_data.shape[1], self.pixelink_image_data.shape[0])
        else:
            return (0, 0, 100, 100)  # Default fallback

    def _setup_results_tab(self):
        """Setup results tab"""
        # Control buttons at top
        control_frame = Frame(self.tab_results, bg=self.DGRAY)
        control_frame.pack(fill=X, padx=5, pady=5)
        
        CButton(control_frame, text="Odśwież", command=self.load_measurements).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Eksportuj wszystkie", command=self.export_measurements).pack(side=LEFT, padx=5)
        CButton(control_frame, text="Usuń wszystkie", command=self.delete_all_measurements).pack(side=LEFT, padx=5)
        
        # Info label
        self.results_info = Label(
            control_frame, 
            text="Pomiary: 0", 
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

        Label(settings_frame, text="Port AUX1:", bg=self.DGRAY, fg='white').grid(row=row_base+3, column=0, sticky=W, pady=5)
        self.port_aux1_var = StringVar(value=options.get('port_aux1', ''))
        self.port_aux1_combo = ttk.Combobox(settings_frame, textvariable=self.port_aux1_var, values=ports)
        self.port_aux1_combo.grid(row=row_base+3, column=1, sticky=EW, pady=5)

        Label(settings_frame, text="Port AUX2:", bg=self.DGRAY, fg='white').grid(row=row_base+4, column=0, sticky=W, pady=5)
        self.port_aux2_var = StringVar(value=options.get('port_aux2', ''))
        self.port_aux2_combo = ttk.Combobox(settings_frame, textvariable=self.port_aux2_var, values=ports)
        self.port_aux2_combo.grid(row=row_base+4, column=1, sticky=EW, pady=5)

        CButton(settings_frame, text="Refresh Ports", command=self.refresh_ports).grid(row=row_base+1, column=2, rowspan=4, padx=10, sticky=N)

        # Camera settings
        cam_row = row_base + 5
        Label(settings_frame, text="Camera Settings", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=cam_row, column=0, columnspan=3, pady=10, sticky=W)
        
        Label(settings_frame, text="Camera Index:", bg=self.DGRAY, fg='white').grid(row=cam_row+1, column=0, sticky=W, pady=5)
        self.camera_index_var = IntVar(value=options.get('camera_index', 0))
        cams = self._list_cameras()
        self.camera_combo = ttk.Combobox(settings_frame, values=cams, state='readonly', width=10)
        try:
            self.camera_combo.set(str(self.camera_index_var.get()) if self.camera_index_var.get() in cams else str(cams[0]))
        except Exception:
            pass
        self.camera_combo.grid(row=cam_row+1, column=1, sticky=EW, pady=5)
        CButton(settings_frame, text="Refresh", command=self.refresh_cameras).grid(row=cam_row+1, column=2, padx=10)
        
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

        # Apply button - make it more prominent
        apply_frame = Frame(settings_frame, bg=self.DGRAY)
        apply_frame.grid(row=calib_row+4, column=0, columnspan=3, pady=20)
        
        CButton(apply_frame, text="💾 SAVE SETTINGS", command=self.apply_settings, 
               font=("Arial", 12, "bold"), fg='yellow').pack(pady=5)
        
        Label(apply_frame, text="Click to save all changes to options.json", 
              bg=self.DGRAY, fg='lightgray', font=("Arial", 9)).pack()
        
        settings_frame.columnconfigure(1, weight=1)

    def refresh_ports(self):
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            for combo in [self.port_x_combo, self.port_y_combo, self.port_aux1_combo, self.port_aux2_combo]:
                combo.configure(values=ports)
            print(f"Ports refreshed: {ports}")
        except Exception as e:
            print(f"Ports refresh error: {e}")

    def _list_cameras(self):
        cameras = []
        try:
            for idx in range(0, 10):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    cameras.append(idx)
                    cap.release()
        except Exception:
            pass
        if not cameras:
            cameras = [0]
        return cameras

    def save_camera_selection(self):
        try:
            idx = int(self.camera_combo.get()) if hasattr(self, 'camera_combo') and self.camera_combo.get() != '' else self.camera_index
            # Update in-memory and file
            self.camera_index = idx
            options['camera_index'] = idx
            with open('options.json', 'w') as f:
                json.dump(options, f, indent=4)
            print(f"Camera selection saved (index={idx})")
        except Exception as e:
            print(f"Camera save error: {e}")

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

    def refresh_cameras(self):
        cams = self._list_cameras()
        self.camera_combo.configure(values=cams)
        try:
            self.camera_combo.current(cams.index(self.camera_index_var.get()) if self.camera_index_var.get() in cams else 0)
        except Exception:
            pass
        print(f"Cameras found: {cams}")

    def _setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.DGRAY, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.DGRAY, foreground='white')
        style.map('TNotebook.Tab', background=[('selected', self.RGRAY)])

    def _load_test_image_data(self):
        """Load test image 2.bmp or create default data"""
        try:
            # Try to load 2.bmp image
            if os.path.exists("2.bmp"):
                image = Image.open("2.bmp")
                
                # Limit image size to prevent memory issues
                max_size = (2048, 1536)  # Max size to prevent memory errors
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    print(f"Resizing large image from {image.size} to {max_size}")
                    image = image.resize(max_size, Image.Resampling.LANCZOS)
                
                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')
                    
                # Convert to numpy array
                image_array = np.array(image)
                print(f"Loaded test image: 2.bmp, size: {image_array.shape}, memory: {image_array.nbytes / 1024:.1f} KB")
                
                # Calculate spectrum safely
                if image_array.size > 0:
                    spectrum = np.mean(image_array, axis=0)
                    self.spectrum_data = self._resample_to_2048(spectrum)
                    self._update_spectrum_plot()
                    
                return image_array
            else:
                print("Test image 2.bmp not found, using default data")
                return np.zeros((100, 100), dtype=np.uint8)
        except Exception as e:
            print(f"Error loading test image: {e}")
            return np.zeros((100, 100), dtype=np.uint8)

    def _style_toolbar(self, toolbar):
        """Style toolbar to match dark theme"""
        toolbar.config(bg=self.DGRAY)
        
        # Style all toolbar children
        for child in toolbar.winfo_children():
            try:
                if hasattr(child, 'config'):
                    child.config(bg=self.DGRAY)
                    
                    # Handle different widget types
                    widget_class = child.winfo_class()
                    
                    if widget_class == 'Button':
                        child.config(
                            bg=self.RGRAY, 
                            fg='white',
                            activebackground=self.MGRAY,
                            activeforeground='white',
                            relief='flat',
                            bd=1
                        )
                    elif widget_class == 'Frame':
                        child.config(bg=self.DGRAY)
                        # Recursively style frame children
                        for subchild in child.winfo_children():
                            try:
                                subchild.config(bg=self.DGRAY, fg='white')
                            except:
                                pass
                    elif widget_class == 'Label':
                        child.config(bg=self.DGRAY, fg='white')
                    
            except Exception as e:
                pass  # Ignore styling errors for unsupported widgets

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
                'błąd', 'error', 'exception', 'failed', 'nie powiodł', 'nie są podłączone',
                'sprawdź połączenie', 'traceback', 'brak wykrytego'
            ])
            
            # Detect warning messages and apply yellow color
            is_warning = any(keyword in message.lower() for keyword in [
                'ostrzeżenie', 'warning', 'uwaga', 'attention', 'simulation', 'symulacja',
                'not connected', 'brak połączenia', 'przybliżona', 'approximate'
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
        
        # Start update loop immediately - don't wait for initialization
        self.update_loop()
        
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
            
            print("Background: Loading measurements...")
            # Load measurements in background
            self.load_measurements()
            
            print("Background: Starting worker threads...")
            # Start background threads after everything is ready
            self._start_background_threads()
            
            self.after_idle(lambda: print("✅ System initialization complete"))
            
        except Exception as e:
            self.after_idle(lambda: print(f"❌ Background initialization error: {e}"))

    def _fallback_to_test_image(self):
        """Fallback to test image when camera initialization fails"""
        if not self.is_live_streaming:
            self.pixelink_status.config(text="Status: No camera - Using test image")
            print("🔸 No Pixelink camera available, using test image instead")
            print("🔸 Make sure Pixelink camera is connected and drivers are installed")
            self.load_test_image()

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        
        try:
            self.camera_manager.stop()
            self.spectrometer_manager.stop()
            self.motor_controller.close()
            # Close aux serials
            if hasattr(self, 'aux_serial'):
                for key, ser in list(self.aux_serial.items()):
                    try:
                        ser.close()
                    except Exception:
                        pass
                self.aux_serial.clear()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        # Restore stdout
        sys.stdout = sys.__stdout__

    def start_camera(self):
        """Start camera"""
        try:
            # Switch camera if selection changed
            selected = None
            try:
                selected = int(self.camera_combo.get()) if hasattr(self, 'camera_combo') else self.camera_index
            except Exception:
                selected = self.camera_index
            if selected is None:
                selected = options.get('camera_index', 0)
            if selected != self.camera_manager.camera_index:
                self.camera_manager.stop()
                self.camera_manager = CameraManager(camera_index=selected)
            self.camera_manager.start()
            self.camera_index = selected
            # Verify camera opened
            opened = False
            try:
                opened = self.camera_manager.detector is not None and self.camera_manager.detector.isOpened()
            except Exception:
                opened = False
            if opened:
                print(f"Camera started (index={selected})")
                self.cam_status.configure(text=f"Pixelink Camera: Running (index={selected})", fg='lightgreen')
            else:
                print("No camera detected or failed to open. Showing placeholder.")
                self.cam_status.configure(text="Pixelink Camera: Failed to start", fg='red')
                self._draw_camera_placeholder()
        except Exception as e:
            print(f"Camera start error: {e}")
            self.cam_status.configure(text=f"Pixelink Camera: Error - {e}", fg='red')
            try:
                self._draw_camera_placeholder()
            except Exception:
                pass
                
    def stop_camera(self):
        """Stop camera"""
        self.camera_manager.stop()
        print("Camera stopped")
        self.cam_status.configure(text="Pixelink Camera: Stopped", fg='orange')
        try:
            self._draw_camera_placeholder()
        except Exception:
            pass

    def _move_and_record(self, direction, steps=None):
        try:
            self._last_move_dir = direction
            self.motor_controller.move(direction, steps)
        except Exception as e:
            print(f"Move error: {e}")

    def on_spectrum_mouse_move(self, event):
        """Handle mouse movement over spectrum plot"""
        if event.inaxes == self.spectrum_ax:
            try:
                # Remove previous crosshair
                if self.spectrum_crosshair_v:
                    self.spectrum_crosshair_v.remove()
                if self.spectrum_crosshair_h:
                    self.spectrum_crosshair_h.remove()
                if self.spectrum_annotation:
                    self.spectrum_annotation.remove()
                
                # Add new crosshair
                self.spectrum_crosshair_v = self.spectrum_ax.axvline(event.xdata, color='red', alpha=0.7, linestyle='--')
                self.spectrum_crosshair_h = self.spectrum_ax.axhline(event.ydata, color='red', alpha=0.7, linestyle='--')
                
                # Add annotation with values
                self.spectrum_annotation = self.spectrum_ax.annotate(
                    f'X: {event.xdata:.1f}\nY: {event.ydata:.1f}',
                    xy=(event.xdata, event.ydata),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=10
                )
                
                self.spectrum_canvas.draw_idle()
                
            except Exception as e:
                pass  # Ignore errors during mouse movement

    def on_spectrum_click(self, event):
        """Handle mouse click on spectrum plot"""
        if event.inaxes == self.spectrum_ax and event.dblclick:
            try:
                # Double-click to mark peak
                x_pos = event.xdata
                y_pos = event.ydata
                
                # Add permanent marker
                self.spectrum_ax.plot(x_pos, y_pos, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                self.spectrum_ax.annotate(
                    f'Peak\n({x_pos:.1f}, {y_pos:.1f})',
                    xy=(x_pos, y_pos),
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                    fontsize=9, color='black'
                )
                
                self.spectrum_canvas.draw()
                print(f"Peak marked at: X={x_pos:.1f}, Y={y_pos:.1f}")
                
            except Exception as e:
                print(f"Peak marking error: {e}")

    def reset_spectrum_view(self):
        """Reset spectrum view to show all data"""
        try:
            self.spectrum_ax.set_xlim(0, len(self.spectrum_data))
            self.spectrum_ax.set_ylim(0, np.max(self.spectrum_data) * 1.1 if np.max(self.spectrum_data) > 0 else 1000)
            
            # Clear annotations and markers
            annotations = [child for child in self.spectrum_ax.get_children() 
                          if hasattr(child, 'get_text') or 
                          (hasattr(child, 'get_marker') and child.get_marker() == 'o')]
            for ann in annotations:
                try:
                    ann.remove()
                except:
                    pass
            
            self.spectrum_canvas.draw()
            print("Spectrum view reset")
            
        except Exception as e:
            print(f"Spectrum view reset error: {e}")

    def auto_scale_spectrum(self):
        """Auto-scale spectrum to fit data"""
        try:
            if len(self.spectrum_data) > 0:
                # Find data range
                y_min = np.min(self.spectrum_data)
                y_max = np.max(self.spectrum_data)
                y_range = y_max - y_min
                
                # Prevent identical ylim values
                if abs(y_range) < 1e-10:  # Values are essentially the same
                    y_center = y_min if y_min != 0 else 1
                    margin = abs(y_center) * 0.1
                    self.spectrum_ax.set_ylim(y_center - margin, y_center + margin)
                else:
                    # Add 10% margin
                    margin = y_range * 0.1
                    self.spectrum_ax.set_ylim(y_min - margin, y_max + margin)
                
                # Auto-scale X to show interesting part (where there's signal)
                threshold = y_min + y_range * 0.1  # 10% above minimum
                signal_indices = np.where(self.spectrum_data > threshold)[0]
                
                if len(signal_indices) > 0:
                    x_start = max(0, signal_indices[0] - 50)
                    x_end = min(len(self.spectrum_data), signal_indices[-1] + 50)
                    self.spectrum_ax.set_xlim(x_start, x_end)
                
                self.spectrum_canvas.draw()
                print(f"Auto-scaled spectrum: Y=[{y_min:.1f}, {y_max:.1f}]")
                
        except Exception as e:
            print(f"Auto-scale error: {e}")

    def set_spectrum_range(self):
        """Set spectrum display range"""
        try:
            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            
            # Update plot range
            self.spectrum_ax.set_xlim(xmin, xmax)
            
            # Add range indicator lines
            if hasattr(self, 'range_lines'):
                for line in self.range_lines:
                    try:
                        line.remove()
                    except:
                        pass
            
            line1 = self.spectrum_ax.axvline(xmin, color='red', linestyle='--', alpha=0.7, linewidth=2)
            line2 = self.spectrum_ax.axvline(xmax, color='red', linestyle='--', alpha=0.7, linewidth=2)
            self.range_lines = [line1, line2]
            
            # Add range labels
            self.spectrum_ax.text(xmin, self.spectrum_ax.get_ylim()[1] * 0.9, f'Min: {xmin}', 
                                 rotation=90, verticalalignment='top', color='red', fontweight='bold')
            self.spectrum_ax.text(xmax, self.spectrum_ax.get_ylim()[1] * 0.9, f'Max: {xmax}', 
                                 rotation=90, verticalalignment='top', color='red', fontweight='bold')
            
            self.spectrum_canvas.draw()
            print(f"Spectrum range set: {xmin} - {xmax}")
            
        except Exception as e:
            print(f"Range setting error: {e}")

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
                
                # Use after_idle to queue canvas update - completely non-blocking
                self.after_idle(self._safe_canvas_update)
                
        except Exception as e:
            print(f"Spectrum plot update error: {e}")

    def _safe_canvas_update(self):
        """Safe canvas update in main thread"""
        try:
            if hasattr(self, 'spectrum_canvas'):
                self.spectrum_canvas.draw_idle()
        except Exception:
            pass  # Ignore canvas errors to prevent freezing

    def load_test_image(self):
        """Load test image button handler"""
        try:
            if os.path.exists("2.bmp"):
                # Stop live streaming when manually loading test image
                self.is_live_streaming = False
                
                image_data = self._load_test_image_data()
                self.update_pixelink_display(image_data)
                self.pixelink_status.config(text="Status: Test image loaded")
                self.spectrum_data = self._resample_to_2048(np.mean(image_data, axis=0))
                self._update_spectrum_plot()  # Update spectrum with test image data
                
                print("Test image loaded successfully")
            else:
                print("Test image 2.bmp not found")
                self.pixelink_status.config(text="Status: Test image not found")
        except Exception as e:
            print(f"Error loading test image: {e}")
            self.pixelink_status.config(text="Status: Test image error")

    def init_pixelink(self):
        """Initialize Pixelink camera"""
        try:
            if self.spectrometer_manager.initialize():
                self.pixelink_status.config(text="Status: Initialized")
                print("Pixelink camera initialized successfully")
            else:
                # If initialization fails and not live streaming, load test image
                if not self.is_live_streaming:
                    self.pixelink_status.config(text="Status: Using test image")
                    print("Pixelink camera initialization failed, loading test image")
                    self.load_test_image()
                else:
                    self.pixelink_status.config(text="Status: Init failed but streaming continues")
                    print("Pixelink camera initialization failed, but live streaming continues")
        except Exception as e:
            # If error and not live streaming, load test image
            if not self.is_live_streaming:
                self.pixelink_status.config(text="Status: Error - Using test image")
                print(f"Pixelink initialization error: {e}, loading test image")
                self.load_test_image()
            else:
                self.pixelink_status.config(text="Status: Error but streaming continues")
                print(f"Pixelink initialization error: {e}, but live streaming continues")

    def start_pixelink(self):
        """Start Pixelink stream"""
        try:
            if hasattr(self.spectrometer_manager, 'hCamera') and self.spectrometer_manager.hCamera:
                self.spectrometer_manager.start()
                self.is_live_streaming = True  # Mark as live streaming
                self.pixelink_status.config(text="Status: Streaming")
                print("Pixelink stream started")
            else:
                # If no camera, simulate with test image updates
                self.is_live_streaming = False  # Not live streaming
                self.pixelink_status.config(text="Status: Simulating with test image")
                print("No camera available, using test image")
                self._start_test_image_simulation()
        except Exception as e:
            self.is_live_streaming = False  # Not live streaming due to error
            self.pixelink_status.config(text="Status: Stream error - Using test image")
            print(f"Pixelink stream error: {e}, using test image")
            self._start_test_image_simulation()

    def _start_test_image_simulation(self):
        """Start test image simulation"""
        def simulate_updates():
            """Simulate camera updates with test image"""
            try:
                # Stop simulation if live streaming started
                if self.is_live_streaming:
                    print("Stopping simulation - live streaming started")
                    return
                    
                # Add some noise to simulate live data
                if hasattr(self, 'pixelink_image_data'):
                    noise = np.random.random(self.pixelink_image_data.shape) * 10
                    simulated_data = self.pixelink_image_data + noise
                    self.update_pixelink_display(simulated_data)
                
                # Schedule next update only if not live streaming
                if not self.is_live_streaming:
                    self.after(100, simulate_updates)  # Update every 100ms
                
            except Exception as e:
                print(f"Simulation error: {e}")
        
        # Start simulation
        simulate_updates()

    def stop_pixelink(self):
        """Stop Pixelink stream"""
        try:
            self.spectrometer_manager.stop()
            self.is_live_streaming = False  # Reset streaming flag
            self.pixelink_status.config(text="Status: Stopped")
            print("Pixelink stream stopped")
        except Exception as e:
            self.is_live_streaming = False  # Reset streaming flag
            print(f"Pixelink stop error: {e}")

    def reset_pixelink_view(self):
        """Reset Pixelink view to default"""
        try:
            self.pixelink_ax.set_xlim(0, self.pixelink_image_data.shape[1])
            self.pixelink_ax.set_ylim(0, self.pixelink_image_data.shape[0])
            self.pixelink_canvas.draw()
            print("Pixelink view reset")
        except Exception as e:
            print(f"View reset error: {e}")

    def update_pixelink_display(self, image_data):
        """Update Pixelink display with new image data"""
        try:
            if image_data is not None:
                # Handle data from spectrometer queue (dict with frame and descriptor)
                if isinstance(image_data, dict) and 'frame' in image_data and 'descriptor' in image_data:
                    frame = image_data['frame']
                    descriptor = image_data['descriptor']
                    
                    # Use full frame buffer without cropping - let PIL handle it
                    self.pixelink_image_data = frame
                else:
                    # Handle direct image data (for compatibility)
                    self.pixelink_image_data = image_data
                
                # Mark as live streaming when receiving real camera data
                if hasattr(self.spectrometer_manager, 'hCamera') and self.spectrometer_manager.hCamera:
                    self.is_live_streaming = True
                
                # NOTE: Image display update is now handled in main update_loop with throttling
                # Only calculate spectrum here for fast updates
                
                # Calculate spectrum from image data
                raw_spectrum = np.mean(self.pixelink_image_data, axis=0)
                self.spectrum_data = self._resample_to_2048(raw_spectrum)
                self._update_spectrum_plot()
                
        except Exception as e:
            print(f"Pixelink display update error: {e}")

    def start_measurement_sequence(self):
        """Start automated measurement sequence"""
        print("🔥 PRZYCISK SEKWENCJI ZOSTAŁ KLIKNIĘTY!")
        
        def sequence():
            try:
                print("🚀 URUCHAMIANIE SEKWENCJI POMIAROWEJ...")
                print(f"Kalibracja: {getattr(self, 'calibration', False)}")
                roi_coords = getattr(self, 'roi_coordinates', None)
                print(f"ROI coordinates: {roi_coords}")
                
                # Guards: require calibration and scan area
                if not getattr(self, 'calibration', False):
                    print("❌ BŁĄD: Kalibracja wymagana. Przeprowadź kalibrację w zakładce Camera & Controls.")
                    return
                if not roi_coords:
                    print("❌ BŁĄD: Obszar skanowania nie został ustawiony. Zaznacz ROI na obrazie kamery.")
                    return
                
                # Check motor connection
                if not self.motor_controller.connected:
                    print("BŁĄD: Silniki nie są podłączone!")
                    print("Sprawdź połączenia portów szeregowych w zakładce Settings.")
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
                    print("BŁĄD: Kamera/spektrometr nie jest podłączony!")
                    print("Sprawdź połączenie urządzenia.")
                    return
                
                # Create data folder
                folder = "pomiar_dane"
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"pomiar_{time.strftime('%Y%m%d_%H%M%S')}_spectra.csv")
                
                # Get ROI bounds and calculate scan parameters
                x, y, w, h = self.get_roi_bounds()
                step_x = self.step_x.get()
                step_y = self.step_y.get()
                
                # Calculate number of scan points
                nx = max(1, (w // step_x) + 1)
                ny = max(1, (h // step_y) + 1)
                total_points = nx * ny
                
                print(f"📐 Obszar skanowania: {w}x{h} px, startując od ({x}, {y})")
                print(f"📊 Krok skanowania: {step_x}x{step_y} px")
                print(f"🎯 Punkty skanowania: {nx} x {ny} = {total_points} punktów")
                
                # Initialize progress tracking
                current_point = 0
                start_time = time.time()
                
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    count = len(self.spectrum_data)
                    writer.writerow(['measurement_id', 'x_pixel', 'y_pixel', 'x_motor', 'y_motor'] + 
                                  [f'wavelength_{i}' for i in range(count)])
                    
                    print("🏁 Rozpoczynam skanowanie kwadratowe od narożnika...")
                    
                    # Use settings from options.json for scanning area
                    scan_step_x = self.step_x.get()
                    scan_step_y = self.step_y.get()
                    scan_width = self.scan_width.get()
                    scan_height = self.scan_height.get()
                    
                    print(f"📋 Parametry skanowania z ustawień:")
                    print(f"   Krok X: {scan_step_x}, Krok Y: {scan_step_y}")
                    print(f"   Szerokość: {scan_width}, Wysokość: {scan_height}")
                    
                    # Calculate number of points in each direction
                    points_x = scan_width // scan_step_x + 1
                    points_y = scan_height // scan_step_y + 1
                    total_points = points_x * points_y
                    
                    print(f"📐 Siatka skanowania: {points_x} x {points_y} = {total_points} punktów")
                    
                    # Calculate offset to move to top-left corner of scan area
                    # From current position, move to corner where scanning will start
                    offset_x = -scan_width // 2
                    offset_y = -scan_height // 2
                    
                    print(f"📍 Przesunięcie do narożnika od obecnego położenia: ({offset_x}, {offset_y})")
                    
                    # Move to top-left corner of scan area
                    if offset_x != 0:
                        dir_x = 'l' if offset_x < 0 else 'r'
                        self.motor_controller.move(dir_x, abs(offset_x))
                    if offset_y != 0:
                        dir_y = 'd' if offset_y < 0 else 'u'
                        self.motor_controller.move(dir_y, abs(offset_y))
                    
                    time.sleep(1)
                    print("✅ W narożniku - gotowy do skanowania")
                    
                    current_point = 0
                    
                    # Main scanning loop - square grid from corner
                    for iy in range(points_y):
                        print(f"📍 Skanowanie rzędu {iy + 1}/{points_y}")
                        
                        for ix in range(points_x):
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
                            
                            # Small delay for data acquisition
                            time.sleep(0.02)
                
                # Return to original position (before scan started)
                print("🔙 Powrót do pozycji sprzed skanowania...")
                
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
                
                print("✅ Powrócono do pozycji sprzed skanowania")
                
                total_time = time.time() - start_time
                print(f"✅ SKANOWANIE ZAKOŃCZONE!")
                print(f"📁 Zapisano {total_points} pomiarów do: {filename}")
                print(f"⏱️ Czas skanowania: {total_time:.1f} sekund")
                print(f"⚡ Średnio {total_time/total_points:.2f} s/punkt")
                
                # Reload measurements
                self.after(100, self.load_measurements)
                
            except Exception as e:
                print(f"Sequence error: {e}")
        
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
            'port_aux1': self.port_aux1_var.get() if hasattr(self, 'port_aux1_var') else options.get('port_aux1', ''),
            'port_aux2': self.port_aux2_var.get() if hasattr(self, 'port_aux2_var') else options.get('port_aux2', ''),
            'camera_index': int(self.camera_combo.get()) if hasattr(self, 'camera_combo') and self.camera_combo.get() != '' else options.get('camera_index', 0),
            'cal_px_per_step_x': float(self.cal_x_var.get()) if hasattr(self, 'cal_x_var') else options.get('cal_px_per_step_x', 0.0),
            'cal_px_per_step_y': float(self.cal_y_var.get()) if hasattr(self, 'cal_y_var') else options.get('cal_px_per_step_y', 0.0),
            'cal_sign_x': int(getattr(self, 'cal_sign_x', options.get('cal_sign_x', 1))),
            'cal_sign_y': int(getattr(self, 'cal_sign_y', options.get('cal_sign_y', 1))),
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

            # (Re)open AUX ports
            try:
                # Close existing
                for key, ser in list(self.aux_serial.items()):
                    try:
                        ser.close()
                    except Exception:
                        pass
                    self.aux_serial.pop(key, None)
                for label, port in [('aux1', settings.get('port_aux1', '')), ('aux2', settings.get('port_aux2', ''))]:
                    if port:
                        try:
                            ser = serial.Serial(port)
                            self.aux_serial[label] = ser
                            print(f"AUX port {label} opened on {port}")
                        except Exception as e:
                            print(f"Failed to open AUX port {label} on {port}: {e}")
            except Exception as e:
                print(f"AUX ports reinit error: {e}")
            
        except Exception as e:
            print(f"Settings save error: {e}")

    def start_calibration(self, axis):
        # This simple manual calibration will store two clicks before/after a known step
        # User flow: Click center on image, press move in axis, click center again, we compute px/step
        if axis not in ('x', 'y'):
            return
        self._calibration_mode = f"{axis}_before"
        print(f"Kalibracja {axis.upper()} — kliknij punkt referencyjny na obrazie.")
        # Temporarily bind a one-off click
        cid = self.pixelink_canvas.mpl_connect('button_press_event', lambda e, ax=axis: self._on_calibration_click(e, ax))
        self._pixelink_event_cids.append(cid)

    def start_calibration_both(self):
        try:
            # Check motor connection first
            if not self.motor_controller.connected:
                print("BŁĄD: Silniki nie są podłączone!")
                print("Sprawdź połączenia portów szeregowych w zakładce Settings.")
                print("Porty X i Y muszą być dostępne do kalibracji.")
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
                print("BŁĄD: Kamera nie jest podłączona i brak obrazu testowego!")
                print("Sprawdź połączenie kamery lub użyj obrazu testowego.")
                return
                
            print("Automatyczna kalibracja: wykonuję ruchy X i Y, analizuję zmiany obrazu...")
            # Clear previous calibration data
            self._calibration_points = {}
            self._cal_images = {}
            self._last_move_dir = None
            self._calibration_mode = 'auto_running'
            
            # Start automatic calibration sequence
            self.after(100, self._auto_calibration_sequence)
        except Exception as e:
            print(f"Auto calibration error: {e}")

    def _auto_calibration_sequence(self):
        """Automatic calibration: move motors, analyze image changes"""
        try:
            if self._calibration_mode == 'auto_running':
                # Step 1: Capture reference image
                print("Krok 1: Zapisuję obraz referencyjny...")
                self._cal_images['reference'] = self.pixelink_image_data.copy()
                
                # Step 2: Move X axis
                print("Krok 2: Wykonuję ruch +X...")
                step_x = max(1, int(options.get('cal_step_x', 1)))
                self.motor_controller.move('r', step_x)
                self._calibration_mode = 'wait_x'
                # Wait for movement to complete and image to update
                self.after(1000, self._auto_calibration_sequence)
                
            elif self._calibration_mode == 'wait_x':
                # Step 3: Capture image after X movement
                print("Krok 3: Analizuję zmianę po ruchu X...")
                self._cal_images['after_x'] = self.pixelink_image_data.copy()
                
                # Analyze X displacement
                self._analyze_x_displacement()
                
                # Step 4: Move Y axis
                print("Krok 4: Wykonuję ruch +Y...")
                step_y = max(1, int(options.get('cal_step_y', 1)))
                self.motor_controller.move('u', step_y)
                self._calibration_mode = 'wait_y'
                # Wait for movement to complete
                self.after(1000, self._auto_calibration_sequence)
                
            elif self._calibration_mode == 'wait_y':
                # Step 5: Capture image after Y movement and finalize
                print("Krok 5: Analizuję zmianę po ruchu Y...")
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
                print(f"X kalibracja: {displacement_x:.1f}px przesunięcie, {self.cal_px_per_step_x:.3f} px/step")
            else:
                print("X: Nie wykryto przesunięcia obrazu")
                
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
                print(f"Y kalibracja: {displacement_y:.1f}px przesunięcie, {self.cal_px_per_step_y:.3f} px/step")
            else:
                print("Y: Nie wykryto przesunięcia obrazu")
                
        except Exception as e:
            print(f"Y analysis error: {e}")
    
    def _finalize_auto_calibration(self):
        """Complete automatic calibration and enable ROI"""
        try:
            # Update calibration displays
            if hasattr(self, 'cal_x_var'):
                self.cal_x_var.set(self.cal_px_per_step_x)
            if hasattr(self, 'cal_y_var'):
                self.cal_y_var.set(self.cal_px_per_step_y)
            
            # Set calibration complete
            self.calibration = bool(self.cal_px_per_step_x > 0 and self.cal_px_per_step_y > 0)
            
            if self.calibration:
                print("✓ Automatyczna kalibracja zakończona pomyślnie!")
                print(f"X: {self.cal_px_per_step_x:.3f} px/step, znak: {self.cal_sign_x}")
                print(f"Y: {self.cal_px_per_step_y:.3f} px/step, znak: {self.cal_sign_y}")
                
                # Enable ROI drawing
                self.roi_enabled = True
                print("✓ ROI selection enabled - you can now select scan area")
                print("✓ Przejdź do zakładki 'Spectrum' i kliknij 'Select ROI' aby zaznaczyć obszar")
                
                # Auto-save calibration
                self.save_calibration()
                
                # Generate and store scan points for visualization
                self.scan_points = self.generate_scan_points()
                print(f"✓ Wygenerowano {len(self.scan_points)} punktów skanowania")
                
                # Auto-set a default ROI if none exists
                if not hasattr(self, 'roi_coordinates') or not self.roi_coordinates:
                    try:
                        h, w = self.pixelink_image_data.shape[:2]
                        # Set ROI to center quarter of image
                        center_x, center_y = w//2, h//2
                        roi_size = min(w, h) // 4
                        x1 = center_x - roi_size//2
                        y1 = center_y - roi_size//2
                        x2 = center_x + roi_size//2
                        y2 = center_y + roi_size//2
                        self.roi_coordinates = (x1, y1, x2, y2)
                        print(f"✓ Ustawiono domyślny ROI: {self.get_roi_bounds()}")
                        print("✓ Możesz zmienić ROI klikając 'Select ROI' w zakładce Spectrum")
                        
                        # Regenerate scan points with new ROI
                        self.scan_points = self.generate_scan_points()
                    except Exception as e:
                        print(f"Nie udało się ustawić domyślnego ROI: {e}")
            else:
                print("BŁĄD: Kalibracja nie powiodła się - brak wykrytego ruchu obrazu")
            
            # Update UI states
            self._update_roi_button_state()  # Add ROI button update
            self._update_start_seq_state()
            self._calibration_mode = None
            
        except Exception as e:
            print(f"Finalization error: {e}")
            self._calibration_mode = None
    
    def generate_scan_points(self):
        """Generate scan points based on current ROI and step settings"""
        if not hasattr(self, 'roi_coordinates') or not self.roi_coordinates:
            return []
            
        # Get ROI bounds and step settings
        x, y, w, h = self.get_roi_bounds()
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
        
        print(f"🎯 Wygenerowano {len(points)} punktów skanowania ({nx}x{ny} siatka)")
        return points
    
    def toggle_scan_points(self):
        """Toggle scan points visualization - simplified for Tkinter"""
        self.show_scan_points = not self.show_scan_points
        print(f"✅ Punkty skanowania: {'WŁĄCZONE' if self.show_scan_points else 'WYŁĄCZONE'} (wizualizacja wyłączona dla wydajności)")
    
    def _update_scan_points_display(self):
        """Update scan points display - disabled for Tkinter performance"""
        print(f"✅ Punkty skanowania: {'WIDOCZNE' if self.show_scan_points else 'UKRYTE'} (wizualizacja wyłączona)")
        pass
        
    # Old guided calibration method - replaced by auto-calibration
    # def _on_calibration_click_guided(self, event): ...

    def _on_calibration_click(self, event, axis):
        if event.inaxes != self.pixelink_ax:
            return
        if self._calibration_mode == f"{axis}_before":
            self._calibration_points[f"{axis}_before"] = (event.xdata, event.ydata)
            self._add_cal_marker(f"{axis}_before", event.xdata, event.ydata, 'yellow' if axis=='x' else 'orange', f"{axis.upper()}₁")
            print(f"Kalibracja {axis.upper()} — zarejestrowano punkt 1 ({event.xdata:.1f}, {event.ydata:.1f}). Wykonaj jeden krok na osi {axis.upper()} i kliknij ten sam punkt.")
            # Enable ROI after user completes first calibration click/move stage
            self.roi_enabled = True
            self._calibration_mode = f"{axis}_after"
        elif self._calibration_mode == f"{axis}_after":
            self._calibration_points[f"{axis}_after"] = (event.xdata, event.ydata)
            self._add_cal_marker(f"{axis}_after", event.xdata, event.ydata, 'lime' if axis=='x' else 'cyan', f"{axis.upper()}₂")
            p0 = self._calibration_points.get(f"{axis}_before")
            p1 = self._calibration_points.get(f"{axis}_after")
            if p0 and p1:
                dx_raw = (p1[0] - p0[0])
                dy_raw = (p1[1] - p0[1])
                if axis == 'x':
                    steps = max(1, int(options.get('cal_step_x', 1)))
                    px_per_step = abs(dx_raw) / steps
                    self.cal_px_per_step_x = px_per_step
                    s_obs = 1 if dx_raw > 0 else (-1 if dx_raw < 0 else getattr(self, 'cal_sign_x', 1))
                    dir_sign = 1 if getattr(self, '_last_move_dir', 'r') == 'r' else (-1 if getattr(self, '_last_move_dir', 'l') == 'l' else 1)
                    self.cal_sign_x = s_obs * dir_sign
                    if hasattr(self, 'cal_x_var'):
                        self.cal_x_var.set(px_per_step)
                    print(f"Kalibracja X zakończona. px/step={px_per_step:.3f}, sign={self.cal_sign_x}")
                else:
                    steps = max(1, int(options.get('cal_step_y', 1)))
                    px_per_step = abs(dy_raw) / steps
                    self.cal_px_per_step_y = px_per_step
                    s_obs = 1 if dy_raw > 0 else (-1 if dy_raw < 0 else getattr(self, 'cal_sign_y', 1))
                    dir_sign = 1 if getattr(self, '_last_move_dir', 'u') == 'u' else (-1 if getattr(self, '_last_move_dir', 'd') == 'd' else 1)
                    self.cal_sign_y = s_obs * dir_sign
                    if hasattr(self, 'cal_y_var'):
                        self.cal_y_var.set(px_per_step)
                    print(f"Kalibracja Y zakończona. px/step={px_per_step:.3f}, sign={self.cal_sign_y}")
            # Set calibration true only if both axes have valid calibration
            try:
                self.calibration = bool(self.cal_px_per_step_x > 0 and self.cal_px_per_step_y > 0)
                if self.calibration:
                    print("Kalibracja na obu osiach zakończona. ROI odblokowane.")
                    self._update_start_seq_state()
            except Exception:
                pass
            self._calibration_mode = None

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
            if hasattr(self, 'pixelink_canvas'):
                self.pixelink_canvas.draw_idle()
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
        pt = self.pixelink_ax.plot([x], [y], marker='o', color=color, markersize=6, linestyle='')[0]
        lbl = self.pixelink_ax.text(x+5, y+5, label, color=color, fontsize=9)
        self._cal_markers[key] = (pt, lbl)
        try:
            self.pixelink_canvas.draw_idle()
        except Exception:
            pass
    
    def load_measurements(self):
        """Load measurement files list without caching data"""
        folder = "pomiar_dane"
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
                        print(f"Pominięto wiersz z błędem: {row} ({e})")
        except Exception as e:
            print(f"Błąd ładowania pliku {filename}: {e}")
        return data
    
    def export_measurements(self):
        """Export all measurements to a single file"""
        if not self.measurement_files:
            messagebox.showinfo("Info", "Brak pomiarów do eksportu")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Eksportuj pomiary",
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
                    
                    print(f"Eksportowano {len(self.measurement_files)} pomiarów do {filename}")
                    messagebox.showinfo("Sukces", f"Pomiary zostały wyeksportowane do:\n{filename}")
                    
            except Exception as e:
                print(f"Błąd eksportu: {e}")
                messagebox.showerror("Błąd", f"Nie można wyeksportować pomiarów:\n{e}")

    def delete_all_measurements(self):
        """Delete all measurements"""
        if not self.measurement_files:
            messagebox.showinfo("Info", "Brak pomiarów do usunięcia")
            return
        
        result = messagebox.askyesno(
            "Usuń wszystkie pomiary",
            f"Czy na pewno chcesz usunąć wszystkie {len(self.measurement_files)} pomiarów?\n"
            "Ta operacja jest nieodwracalna!"
        )
        
        if result:
            try:
                folder = "pomiar_dane"
                deleted_count = 0
                
                # Delete all CSV files
                for filename in glob.glob(os.path.join(folder, "*_spectra.csv")):
                    if os.path.exists(filename):
                        os.remove(filename)
                        deleted_count += 1
                
                self.measurement_files.clear()  # Clear file list, not data cache
                self.draw_measurements()
                
                print(f"Usunięto {deleted_count} plików pomiarów")
                messagebox.showinfo("Sukces", f"Usunięto {deleted_count} pomiarów")
                
            except Exception as e:
                print(f"Błąd usuwania pomiarów: {e}")
                messagebox.showerror("Błąd", f"Nie można usunąć pomiarów:\n{e}")

    def delete_measurement(self, measurement_index):
        """Delete selected measurement"""
        if 0 <= measurement_index < len(self.measurement_files):
            result = messagebox.askyesno(
                "Usuń pomiar",
                f"Czy na pewno chcesz usunąć pomiar {measurement_index + 1}?\n"
                "Ta operacja jest nieodwracalna!"
            )
            
            if result:
                try:
                    # Delete the specific file
                    file_to_delete = self.measurement_files[measurement_index]
                    os.remove(file_to_delete)
                    print(f"Usunięto plik: {os.path.basename(file_to_delete)}")
                    
                    # Remove from file list and refresh
                    self.measurement_files.pop(measurement_index)
                    self.draw_measurements()
                    
                except Exception as e:
                    print(f"Błąd usuwania pomiaru: {e}")
                    messagebox.showerror("Błąd", f"Nie można usunąć pomiaru:\n{e}")

    def draw_measurements(self):
        """Draw measurement buttons in grid layout"""
        # Clear existing buttons
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.measurement_files:
            # Show message if no measurements
            Label(
                self.results_frame, 
                text="Brak pomiarów\nUruchom sekwencję pomiarową aby utworzyć dane",
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

    def show_measurement(self, measurement):
        """Show selected measurement in heatmap window (legacy method - deprecated)"""
        print("Warning: show_measurement is deprecated, use show_measurement_by_index instead")

    def update_loop(self):
        """Ultra-lightweight GUI update loop - absolutely minimal operations"""
        try:
            # Only update spectrum plot if data changed (flag-based) and objects exist
            if (hasattr(self, '_spectrum_needs_update') and 
                self._spectrum_needs_update and 
                hasattr(self, 'spectrum_ax')):
                
                self._update_spectrum_plot()
                self._spectrum_needs_update = False

        except Exception as e:
            # Log errors but don't let them stop the loop
            try:
                print(f"Update loop error: {e}")
            except:
                pass

        # Schedule next update - stable frequency
        try:
            self.after(50, self.update_loop)  # 20 FPS - stable and responsive
        except:
            pass  # Even scheduling errors shouldn't crash the app

    def _resample_to_2048(self, y):
        try:
            if y is None:
                return np.zeros(2048)
            y = np.asarray(y).astype(float)
            n = y.shape[0]
            if n == 2048:
                return y
            x_src = np.linspace(0, 1, n)
            x_dst = np.linspace(0, 1, 2048)
            return np.interp(x_dst, x_src, y)
        except Exception:
            return np.zeros(2048)

    def _update_camera_display(self, frame):
        """Update camera display"""
        try:
            # Resize frame to fit canvas
            cw, ch = self._camera_frame_size
            frame_resized = cv2.resize(frame, (cw, ch))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            # Draw on canvas
            self._camera_canvas_img = photo
            self.camera_canvas.delete('all')
            self.camera_canvas.create_image(0, 0, anchor='nw', image=photo)
            # Crosshair at center
            cx, cy = cw // 2, ch // 2
            self.camera_canvas.create_line(cx-20, cy, cx+20, cy, fill='gray')
            self.camera_canvas.create_line(cx, cy-20, cx, cy+20, fill='gray')
            # Re-draw selection rectangle if exists
            if self.cam_scan_area_norm:
                x0, y0, x1, y1 = self.cam_scan_area_norm
                rx0, ry0 = int(x0 * cw), int(y0 * ch)
                rx1, ry1 = int(x1 * cw), int(y1 * ch)
                self.camera_canvas.create_rectangle(rx0, ry0, rx1, ry1, outline='cyan', width=2)
            
        except Exception as e:
            print(f"Camera display error: {e}")

    def _draw_camera_placeholder(self):
        try:
            cw, ch = self._camera_frame_size
            self.camera_canvas.delete('all')
            # Center crosshair
            cx, cy = cw // 2, ch // 2
            self.camera_canvas.create_rectangle(0, 0, cw, ch, outline=self.DGRAY, width=1)
            self.camera_canvas.create_line(cx-20, cy, cx+20, cy, fill='gray')
            self.camera_canvas.create_line(cx, cy-20, cx, cy+20, fill='gray')
            self.camera_canvas.create_text(cw//2, 30, text="No camera. Select and Start.", fill='lightgray')
        except Exception:
            pass

    def _cam_on_press(self, event):
        self.cam_drag_start = (event.x, event.y)
        if self.cam_drag_rect is not None:
            self.camera_canvas.delete(self.cam_drag_rect)
            self.cam_drag_rect = None

    def _cam_on_drag(self, event):
        if not self.cam_drag_start:
            return
        x0, y0 = self.cam_drag_start
        x1, y1 = event.x, event.y
        if self.cam_drag_rect is not None:
            self.camera_canvas.coords(self.cam_drag_rect, x0, y0, x1, y1)
        else:
            self.cam_drag_rect = self.camera_canvas.create_rectangle(x0, y0, x1, y1, outline='cyan', width=2)

    def _cam_on_release(self, event):
        if not self.cam_drag_start:
            return
        x0, y0 = self.cam_drag_start
        x1, y1 = event.x, event.y
        self.cam_drag_start = None
        # Normalize to 0..1
        cw, ch = self._camera_frame_size
        nx0 = max(0.0, min(1.0, min(x0, x1) / cw))
        ny0 = max(0.0, min(1.0, min(y0, y1) / ch))
        nx1 = max(0.0, min(1.0, max(x0, x1) / cw))
        ny1 = max(0.0, min(1.0, max(y0, y1) / ch))
        self.cam_scan_area_norm = (nx0, ny0, nx1, ny1)
        self.cam_status.config(text=f"Scan area: ({nx0:.2f}, {ny0:.2f})–({nx1:.2f}, {ny1:.2f})")

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

    def _start_background_threads(self):
        """Start background threads for non-blocking data processing"""
        import threading
        
        # Thread for Pixelink data processing
        if hasattr(self, 'spectrometer_manager') and self.spectrometer_manager:
            threading.Thread(target=self._pixelink_data_worker, daemon=True).start()
        
        # Thread for camera data processing  
        if hasattr(self, 'camera_manager') and self.camera_manager:
            threading.Thread(target=self._camera_data_worker, daemon=True).start()

    def _pixelink_data_worker(self):
        """Background thread for processing Pixelink data - completely non-blocking"""
        import time
        while not getattr(self, '_stop_threads', True):
            try:
                # Wait for manager to be ready
                if not hasattr(self, 'spectrometer_manager') or not self.spectrometer_manager:
                    time.sleep(0.1)
                    continue
                    
                if getattr(self.spectrometer_manager, 'running', False):
                    # Absolutely non-blocking queue check
                    try:
                        image_data = self.spectrometer_manager.data_queue.get_nowait()
                        if image_data is not None:
                            # Process in background thread
                            try:
                                with self._data_lock:
                                    self.pixelink_image_data = image_data['frame']
                                    self.is_live_streaming = True
                                    
                                    # Calculate spectrum - check ROI size to prevent memory issues
                                    x, y, w, h = self.get_roi_bounds()
                                    
                                    # Limit ROI size to prevent memory errors
                                    max_roi_height = 500
                                    max_roi_width = 2048
                                    
                                    # Clip ROI to image bounds and memory limits
                                    img_h, img_w = self.pixelink_image_data.shape[:2]
                                    x = max(0, min(x, img_w - 1))
                                    y = max(0, min(y, img_h - 1))
                                    w = min(w, max_roi_width, img_w - x)
                                    h = min(h, max_roi_height, img_h - y)
                                    
                                    roi = self.pixelink_image_data[y:y+h, x:x+w]
                                    if roi.size > 0 and roi.size < 1000000:  # Max 1M pixels
                                        roi_mean = np.mean(roi, axis=0)
                                        self.spectrum_data = self._resample_to_2048(roi_mean)
                                        self._spectrum_needs_update = True
                            except Exception as e:
                                print(f"Pixelink data processing error: {e}")
                                        
                            # Update status (thread-safe, non-blocking)
                            try:
                                self.after_idle(lambda: self.pixelink_status.configure(
                                    text="Spectrum calculation active\nLive data from Pixelink", fg='lightgreen'))
                            except:
                                pass
                    except:
                        pass  # Queue empty - completely normal
                        
                # Process static image data if available
                elif hasattr(self, 'pixelink_image_data') and self.pixelink_image_data is not None:
                    try:
                        with self._data_lock:
                            # Same memory protection as above
                            x, y, w, h = self.get_roi_bounds()
                            
                            # Limit ROI size to prevent memory errors
                            max_roi_height = 500
                            max_roi_width = 2048
                            
                            # Clip ROI to image bounds and memory limits
                            img_h, img_w = self.pixelink_image_data.shape[:2]
                            x = max(0, min(x, img_w - 1))
                            y = max(0, min(y, img_h - 1))
                            w = min(w, max_roi_width, img_w - x)
                            h = min(h, max_roi_height, img_h - y)
                            
                            roi = self.pixelink_image_data[y:y+h, x:x+w]
                            if roi.size > 0 and roi.size < 1000000:  # Max 1M pixels
                                roi_mean = np.mean(roi, axis=0)
                                self.spectrum_data = self._resample_to_2048(roi_mean)
                                self._spectrum_needs_update = True
                    except Exception as e:
                        print(f"Static data processing error: {e}")
                            
            except Exception as e:
                print(f"Pixelink worker error: {e}")
                
            time.sleep(0.1)  # Even slower - 10 FPS processing for stability

    def _camera_data_worker(self):
        """Background thread for processing camera data - completely non-blocking"""
        import time
        while not getattr(self, '_stop_threads', True):
            try:
                # Wait for manager to be ready
                if not hasattr(self, 'camera_manager') or not self.camera_manager:
                    time.sleep(0.1)
                    continue
                    
                if getattr(self.camera_manager, 'running', False):
                    # Process frame queue - absolutely non-blocking
                    try:
                        frame = self.camera_manager.frame_queue.get_nowait()
                        if frame is not None:
                            # Update UI from main thread - non-blocking
                            try:
                                self.after_idle(lambda f=frame: self._update_camera_display(f))
                            except:
                                pass  # Ignore UI update errors
                    except:
                        pass  # Queue empty - completely normal
                        
                    # Process direction queue - absolutely non-blocking
                    try:
                        direction = self.camera_manager.direction_queue.get_nowait()
                        if direction:
                            print(f"Movement detected: {direction}")
                    except:
                        pass  # Queue empty - completely normal
                        
            except Exception as e:
                print(f"Camera worker error: {e}")
                
            time.sleep(0.1)  # Even slower - 10 FPS processing for stability

    def cleanup(self):
        """Cleanup resources when closing application"""
        self._stop_threads = True
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop()
        if hasattr(self, 'spectrometer_manager'):
            self.spectrometer_manager.stop()
        if hasattr(self, 'motor_controller'):
            self.motor_controller.close()


if __name__ == "__main__":
    app = SpektrometerApp()
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        app.cleanup()
