"""
Spektrometr Application - Main File
Refactored with proper threading and structure
"""

import asyncio
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
        'port_x': 'COM5', 'port_y': 'COM9'
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
        """Initialize spectrometer camera"""
        try:
            ret = PxLApi.initialize(0)
            if PxLApi.apiSuccess(ret[0]):
                self.hCamera = ret[1]
                ret = PxLApi.loadSettings(self.hCamera, PxLApi.Settings.SETTINGS_FACTORY)
                if PxLApi.apiSuccess(ret[0]):
                    print("Spectrometer initialized successfully")
                    ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)
                    return True
            return False
        except Exception as e:
            print(f"Spectrometer initialization error: {e}")
            return False
    
    def start(self):
        """Start spectrometer data collection"""
        if self.hCamera and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._spectrometer_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop spectrometer"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.hCamera:
            try:
                PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
                PxLApi.uninitialize(self.hCamera)
            except:
                pass
    
    def _spectrometer_loop(self):
        """Spectrometer data collection loop"""
        while self.running:
            try:
                # Replace with actual PxLApi calls
                spectrum_data = np.random.random(2048) * 1000  # Mock data
                self.data_queue.put(spectrum_data)
                time.sleep(0.1)
            except Exception as e:
                print(f"Spectrometer error: {e}")
                time.sleep(0.1)


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
        
        # Initialize managers
        self.camera_manager = CameraManager()
        self.spectrometer_manager = SpectrometerManager()
        self.motor_controller = MotorController(
            options.get('port_x', 'COM5'),
            options.get('port_y', 'COM9')
        )
        
        # Variables
        self.measurements = []
        self.current_image = None
        self.spectrum_data = np.zeros(2048)
        
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
        """Setup combined camera and controls tab"""
        # Create main container with two columns
        main_container = Frame(self.tab_camera_controls, bg=self.DGRAY)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Left side - Camera
        camera_frame = Frame(main_container, bg=self.DGRAY)
        camera_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        # Camera title
        Label(camera_frame, text="Camera Feed", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack(pady=(0, 10))
        
        # Camera display
        self.camera_label = Label(camera_frame, bg=self.DGRAY, text="Camera feed will appear here", fg='white')
        self.camera_label.pack(fill=BOTH, expand=True)
        
        # Camera control buttons
        camera_control_frame = Frame(camera_frame, bg=self.DGRAY)
        camera_control_frame.pack(fill=X, pady=5)
        
        CButton(camera_control_frame, text="Start Camera", command=self.start_camera).pack(side=LEFT, padx=5)
        CButton(camera_control_frame, text="Stop Camera", command=self.stop_camera).pack(side=LEFT, padx=5)
        
        # Right side - Controls
        controls_frame = Frame(main_container, bg=self.DGRAY)
        controls_frame.pack(side=RIGHT, fill=Y, padx=(5, 0))
        controls_frame.configure(width=400)  # Fixed width for controls
        
        # Controls title
        Label(controls_frame, text="Motor Controls", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').pack(pady=(0, 10))
        
        # Manual controls frame
        manual_frame = LabelFrame(controls_frame, text="Manual Movement", bg=self.DGRAY, fg='white')
        manual_frame.pack(fill=X, pady=10)
        
        # Direction buttons
        button_frame = Frame(manual_frame, bg=self.DGRAY)
        button_frame.pack(pady=10)
        
        CButton(button_frame, text="↑", command=lambda: self.motor_controller.move('u')).grid(row=0, column=1, padx=5, pady=5)
        CButton(button_frame, text="←", command=lambda: self.motor_controller.move('l')).grid(row=1, column=0, padx=5, pady=5)
        CButton(button_frame, text="Origin", command=lambda: self.motor_controller.move('o')).grid(row=1, column=1, padx=5, pady=5)
        CButton(button_frame, text="→", command=lambda: self.motor_controller.move('r')).grid(row=1, column=2, padx=5, pady=5)
        CButton(button_frame, text="↓", command=lambda: self.motor_controller.move('d')).grid(row=2, column=1, padx=5, pady=5)
        
        # Sequence controls
        sequence_frame = LabelFrame(controls_frame, text="Measurement Sequence", bg=self.DGRAY, fg='white')
        sequence_frame.pack(fill=X, pady=10)
        
        CButton(sequence_frame, text="Start Sequence", command=self.start_measurement_sequence).pack(pady=10)
        
        # Console output
        console_frame = LabelFrame(controls_frame, text="Console Output", bg=self.DGRAY, fg='white')
        console_frame.pack(fill=BOTH, expand=True, pady=10)
        
        self.console = Text(console_frame, bg=self.DGRAY, fg='lightgreen', height=10)
        console_scrollbar = Scrollbar(console_frame, orient=VERTICAL, command=self.console.yview)
        self.console.configure(yscrollcommand=console_scrollbar.set)
        
        self.console.pack(side=LEFT, fill=BOTH, expand=True)
        console_scrollbar.pack(side=RIGHT, fill=Y)

    def _setup_spectrum_pixelink_tab(self):
        """Setup combined spectrum and pixelink tab"""
        # Create main container with two sections (top/bottom)
        main_container = Frame(self.tab_spectrum_pixelink, bg=self.DGRAY)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Spectrum
        spectrum_frame = Frame(main_container, bg=self.DGRAY)
        spectrum_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
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
        
        # Bottom section - Pixelink
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
        self.spectrum_fig, self.spectrum_ax = plt.subplots(figsize=(12, 4), facecolor=self.DGRAY)
        self.spectrum_ax.set_facecolor(self.DGRAY)
        
        self.x_axis = np.linspace(0, 2048, 2048)
        self.spectrum_line, = self.spectrum_ax.plot(self.x_axis, self.spectrum_data, color='green', linewidth=1.5)
        
        # Style the plot
        self.spectrum_ax.set_xlabel("Pixel/Wavelength", color='white', fontsize=12)
        self.spectrum_ax.set_ylabel("Intensity", color='white', fontsize=12)
        self.spectrum_ax.set_title("Live Spectrum", color='white', fontsize=14)
        self.spectrum_ax.tick_params(colors='white')
        self.spectrum_ax.grid(True, alpha=0.3, color='gray')
        
        # Set better margins
        self.spectrum_fig.tight_layout()
        
        # Create canvas
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=parent_frame)
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
        
        # Pack canvas
        self.spectrum_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Add mouse interaction
        self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
        self.spectrum_canvas.mpl_connect('button_press_event', self.on_spectrum_click)
        
        # Initialize crosshair
        self.spectrum_crosshair_v = None
        self.spectrum_crosshair_h = None
        self.spectrum_annotation = None

    def _create_pixelink_plot(self, parent_frame):
        """Create pixelink matplotlib plot with navigation toolbar"""
        self.pixelink_fig, self.pixelink_ax = plt.subplots(figsize=(12, 4), facecolor=self.DGRAY)
        self.pixelink_ax.set_facecolor(self.DGRAY)
        
        # Try to load test image first
        self.pixelink_image_data = self._load_test_image_data()
        
        self.pixelink_im = self.pixelink_ax.imshow(
            self.pixelink_image_data, 
            cmap='hot', 
            interpolation='nearest',
            aspect='auto'
        )
        
        # Style the plot
        self.pixelink_ax.set_title("Spectrometer Camera View", color='white', fontsize=14)
        self.pixelink_ax.set_xlabel("X Position", color='white')
        self.pixelink_ax.set_ylabel("Y Position", color='white')
        self.pixelink_ax.tick_params(colors='white')
        
        # Add colorbar
        cbar = self.pixelink_fig.colorbar(self.pixelink_im, ax=self.pixelink_ax)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.ax.yaxis.set_label_text('Intensity', color='white')

        # Create canvas
        self.pixelink_canvas = FigureCanvasTkAgg(self.pixelink_fig, master=parent_frame)
        self.pixelink_canvas.draw()
        
        # Navigation toolbar for zoom/pan
        pixelink_toolbar_frame = Frame(parent_frame, bg=self.DGRAY)
        pixelink_toolbar_frame.pack(fill=X, pady=2)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.pixelink_toolbar = NavigationToolbar2Tk(self.pixelink_canvas, pixelink_toolbar_frame)
        self.pixelink_toolbar.config(bg=self.DGRAY)
        
        # Style toolbar
        self._style_toolbar(self.pixelink_toolbar)
        self.pixelink_toolbar.update()
        
        # Pack canvas
        self.pixelink_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

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
        
        # Port settings
        Label(settings_frame, text="Port Settings", font=("Arial", 14, "bold"), 
              bg=self.DGRAY, fg='white').grid(row=len(settings_data)+2, column=0, columnspan=2, pady=10)
        
        ports = [p.device for p in serial.tools.list_ports.comports()]
        
        Label(settings_frame, text="Port X:", bg=self.DGRAY, fg='white').grid(row=len(settings_data)+3, column=0, sticky=W, pady=5)
        self.port_x_var = StringVar(value=options.get('port_x', 'COM5'))
        ttk.Combobox(settings_frame, textvariable=self.port_x_var, values=ports).grid(row=len(settings_data)+3, column=1, sticky=EW, pady=5)
        
        Label(settings_frame, text="Port Y:", bg=self.DGRAY, fg='white').grid(row=len(settings_data)+4, column=0, sticky=W, pady=5)
        self.port_y_var = StringVar(value=options.get('port_y', 'COM9'))
        ttk.Combobox(settings_frame, textvariable=self.port_y_var, values=ports).grid(row=len(settings_data)+4, column=1, sticky=EW, pady=5)
        
        # Apply button
        CButton(settings_frame, text="Apply Settings", command=self.apply_settings).grid(row=len(settings_data)+6, column=0, columnspan=2, pady=20)
        
        settings_frame.columnconfigure(1, weight=1)

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
                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')
                # Convert to numpy array
                image_array = np.array(image)
                self.spectrum_data = np.mean(image_array, axis=0)  # Example processing
                self._update_spectrum_plot()
                print("Loaded test image: 2.bmp")
                return image_array
            else:
                print("Test image 2.bmp not found, using default data")
                return np.zeros((100, 100))
        except Exception as e:
            print(f"Error loading test image: {e}")
            return np.zeros((100, 100))

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
        """Output text to console Text widget"""
        readable_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        self.console.insert(END, f'{readable_time}: {message}\n')
        self.console.see(END)

    def _delayed_init(self):
        """Initialize systems after GUI is ready"""
        # Initialize spectrometer
        if self.spectrometer_manager.initialize():
            self.spectrometer_manager.start()
            print("Spectrometer initialized successfully")
        else:
            print("Spectrometer initialization failed")
        
        # Load existing measurements
        self.load_measurements()
        
        # Start update loop
        self.update_loop()
        
        print("System initialization complete")

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        
        try:
            self.camera_manager.stop()
            self.spectrometer_manager.stop()
            self.motor_controller.close()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        # Restore stdout
        sys.stdout = sys.__stdout__

    def start_camera(self):
        """Start camera"""
        self.camera_manager.start()
        print("Camera started")

    def stop_camera(self):
        """Stop camera"""
        self.camera_manager.stop()
        print("Camera stopped")

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
        """Update spectrum plot with enhanced features"""
        try:
            if len(self.spectrum_data) > 0:
                # Update main spectrum line
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
                        self.spectrum_ax.set_ylim(data_min * 0.9, data_max * 1.1)
                
                # Update canvas
                self.spectrum_canvas.draw_idle()
                
        except Exception as e:
            print(f"Spectrum plot update error: {e}")

    def load_test_image(self):
        """Load test image button handler"""
        try:
            if os.path.exists("2.bmp"):
                image_data = self._load_test_image_data()
                self.update_pixelink_display(image_data)
                self.pixelink_status.config(text="Status: Test image loaded")
                self.spectrum_data = np.mean(image_data, axis=0)  # Simulate spectrum from image
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
                # If initialization fails, load test image
                self.pixelink_status.config(text="Status: Using test image")
                print("Pixelink camera initialization failed, loading test image")
                self.load_test_image()
        except Exception as e:
            self.pixelink_status.config(text="Status: Error - Using test image")
            print(f"Pixelink initialization error: {e}, loading test image")
            self.load_test_image()

    def start_pixelink(self):
        """Start Pixelink stream"""
        try:
            if hasattr(self.spectrometer_manager, 'hCamera') and self.spectrometer_manager.hCamera:
                self.spectrometer_manager.start()
                self.pixelink_status.config(text="Status: Streaming")
                print("Pixelink stream started")
            else:
                # If no camera, simulate with test image updates
                self.pixelink_status.config(text="Status: Simulating with test image")
                print("No camera available, using test image")
                self._start_test_image_simulation()
        except Exception as e:
            self.pixelink_status.config(text="Status: Stream error - Using test image")
            print(f"Pixelink stream error: {e}, using test image")
            self._start_test_image_simulation()

    def _start_test_image_simulation(self):
        """Start test image simulation"""
        def simulate_updates():
            """Simulate camera updates with test image"""
            try:
                # Add some noise to simulate live data
                if hasattr(self, 'pixelink_image_data'):
                    noise = np.random.random(self.pixelink_image_data.shape) * 10
                    simulated_data = self.pixelink_image_data + noise
                    self.update_pixelink_display(simulated_data)
                
                # Schedule next update
                self.after(100, simulate_updates)  # Update every 100ms
                
            except Exception as e:
                print(f"Simulation error: {e}")
        
        # Start simulation
        simulate_updates()

    def stop_pixelink(self):
        """Stop Pixelink stream"""
        try:
            self.spectrometer_manager.stop()
            self.pixelink_status.config(text="Status: Stopped")
            print("Pixelink stream stopped")
        except Exception as e:
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
                self.pixelink_image_data = image_data
                self.pixelink_im.set_array(image_data)
                self.pixelink_im.set_clim(vmin=np.min(image_data), vmax=np.max(image_data))
                self.pixelink_canvas.draw_idle()
        except Exception as e:
            print(f"Pixelink display update error: {e}")

    def start_measurement_sequence(self):
        """Start automated measurement sequence"""
        def sequence():
            try:
                print("Starting measurement sequence...")
                
                # Create data folder
                folder = "pomiar_dane"
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"pomiar_{time.strftime('%Y%m%d_%H%M%S')}_spectra.csv")
                
                # Get settings
                width = self.scan_width.get()
                height = self.scan_height.get()
                step_x = self.step_x.get()
                step_y = self.step_y.get()
                
                frame_count = 0
                
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['measurement_id', 'x', 'y'] + [f'wavelength_{i}' for i in range(2048)])
                    
                    # Move to start position
                    self.motor_controller.move('o')  # Go to origin
                    time.sleep(1)
                    
                    for x in range(0, width, step_x):
                        for y in range(0, height, step_y):
                            # Move to position
                            if frame_count > 0:
                                self.motor_controller.move('u', step_y)
                                time.sleep(0.1)
                            
                            # Get spectrum data
                            spectrum = self.spectrum_data.copy()
                            xmin = int(self.xmin_var.get())
                            xmax = int(self.xmax_var.get())
                            spectrum_roi = spectrum[xmin:xmax]
                            
                            # Save data
                            writer.writerow([1, x, y] + spectrum_roi.tolist())
                            frame_count += 1
                            
                            print(f"Frame {frame_count}: x={x}, y={y}")
                        
                        # Move to next column
                        if x < width - step_x:
                            self.motor_controller.move('r', step_x)
                            self.motor_controller.move('d', height)
                            time.sleep(0.2)
                
                print(f"Sequence completed! {frame_count} frames saved to {filename}")
                
                # Reload measurements
                self.after(100, self.load_measurements)
                
            except Exception as e:
                print(f"Sequence error: {e}")
        
        # Run sequence in separate thread
        if self.motor_controller.connected:
            threading.Thread(target=sequence, daemon=True).start()
        else:
            print("Motor controller not connected")

    def apply_settings(self):
        """Apply and save settings"""
        settings = {
            'step_x': self.step_x.get(),
            'step_y': self.step_y.get(),
            'width': self.scan_width.get(),
            'height': self.scan_height.get(),
            'xmin': self.xmin_var.get(),
            'xmax': self.xmax_var.get(),
            'port_x': self.port_x_var.get(),
            'port_y': self.port_y_var.get(),
            'await': 0.01
        }
        
        try:
            with open('options.json', 'w') as f:
                json.dump(settings, f, indent=4)
            print("Settings saved successfully")
            
            # Reinitialize motor controller with new ports
            self.motor_controller.close()
            self.motor_controller = MotorController(
                self.port_x_var.get(),
                self.port_y_var.get()
            )
            
        except Exception as e:
            print(f"Settings save error: {e}")
    
    def load_measurements(self):
        """Load measurement files and create result buttons"""
        folder = "pomiar_dane"
        self.measurements = []
        if not os.path.exists(folder):
            os.makedirs(folder)
        for filename in sorted(glob.glob(os.path.join(folder, "*_spectra.csv"))):
            data = []
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
            self.measurements.append(data)   
        self.draw_measurements()
    
    def export_measurements(self):
        """Export all measurements to a single file"""
        if not self.measurements:
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
                    
                    # Write all measurements
                    for measurement_id, measurement in enumerate(self.measurements):
                        for point in measurement:
                            x, y, spectrum = point
                            writer.writerow([measurement_id + 1, x, y] + spectrum)
                    
                    print(f"Eksportowano {len(self.measurements)} pomiarów do {filename}")
                    messagebox.showinfo("Sukces", f"Pomiary zostały wyeksportowane do:\n{filename}")
                    
            except Exception as e:
                print(f"Błąd eksportu: {e}")
                messagebox.showerror("Błąd", f"Nie można wyeksportować pomiarów:\n{e}")

    def delete_all_measurements(self):
        """Delete all measurements"""
        if not self.measurements:
            messagebox.showinfo("Info", "Brak pomiarów do usunięcia")
            return
        
        result = messagebox.askyesno(
            "Usuń wszystkie pomiary",
            f"Czy na pewno chcesz usunąć wszystkie {len(self.measurements)} pomiarów?\n"
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
                
                self.measurements.clear()
                self.draw_measurements()
                
                print(f"Usunięto {deleted_count} plików pomiarów")
                messagebox.showinfo("Sukces", f"Usunięto {deleted_count} pomiarów")
                
            except Exception as e:
                print(f"Błąd usuwania pomiarów: {e}")
                messagebox.showerror("Błąd", f"Nie można usunąć pomiarów:\n{e}")

    def delete_measurement(self, measurement_index):
        """Delete selected measurement"""
        if 0 <= measurement_index < len(self.measurements):
            result = messagebox.askyesno(
                "Usuń pomiar",
                f"Czy na pewno chcesz usunąć pomiar {measurement_index + 1}?\n"
                "Ta operacja jest nieodwracalna!"
            )
            
            if result:
                try:
                    # Find and delete corresponding file
                    folder = "pomiar_dane"
                    files = sorted(glob.glob(os.path.join(folder, "*_spectra.csv")))
                    
                    if measurement_index < len(files):
                        file_to_delete = files[measurement_index]
                        os.remove(file_to_delete)
                        print(f"Usunięto plik: {os.path.basename(file_to_delete)}")
                    
                    # Remove from list and refresh
                    self.measurements.pop(measurement_index)
                    self.draw_measurements()
                    
                except Exception as e:
                    print(f"Błąd usuwania pomiaru: {e}")
                    messagebox.showerror("Błąd", f"Nie można usunąć pomiaru:\n{e}")

    def draw_measurements(self):
        """Draw measurement buttons in grid layout"""
        # Clear existing buttons
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.measurements:
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
            
            for i, measurement in enumerate(self.measurements):
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
                
                # Info label with measurement details
                info_label = Label(
                    button_frame,
                    text=f"{len(measurement)} punktów",
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
            self.results_info.config(text=f"Pomiary: {len(self.measurements)}")

    def show_measurement_by_index(self, measurement_index):
        """Show selected measurement by index"""
        if 0 <= measurement_index < len(self.measurements):
            measurement = self.measurements[measurement_index]
            HeatMapWindow(self, measurement_index + 1, measurement, self.current_image)

    def show_measurement(self, measurement):
        """Show selected measurement in heatmap window (legacy method)"""
        if measurement and measurement in self.measurements:
            i = self.measurements.index(measurement)
            HeatMapWindow(self, i+1, measurement, self.current_image)

    def update_loop(self):
        """Main update loop for GUI"""
        try:
            # Update camera feed
            self.spectrum_data = np.mean(self.pixelink_image_data, axis=0)
            if self.camera_manager.running:
                frame = self.camera_manager.frame_queue.get(timeout=0.01)
                if frame is not None:
                    self._update_camera_display(frame)
                
                direction = self.camera_manager.direction_queue.get(timeout=0.01)
                if direction:
                    print(f"Movement detected: {direction}")
            
            # Update spectrometer data
            if self.spectrometer_manager.running:
                spectrum = self.spectrometer_manager.data_queue.get(timeout=0.01)
                if spectrum is not None:
                    self.spectrum_data = spectrum
                    self._update_spectrum_plot()
                    
                    # Update Pixelink display with spectrum as 2D image
                    if hasattr(self, 'pixelink_canvas'):
                        try:
                            spectrum_2d = spectrum.reshape(64, 32)  # Przykładowy reshape
                            self.update_pixelink_display(spectrum_2d)
                        except:
                            pass  # Ignore reshape errors
            
        except Exception as e:
            pass  # Ignore timeout exceptions
        
        # Schedule next update
        self.after(33, self.update_loop)  # ~30 FPS

    def _update_camera_display(self, frame):
        """Update camera display"""
        try:
            # Resize frame to fit display
            display_width = 800
            display_height = 600
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert to RGB and create PhotoImage
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Camera display error: {e}")


async def async_gui_loop(app):
    """Async wrapper for GUI updates"""
    while True:
        app.update()
        await asyncio.sleep(0.01)  # ~100 FPS GUI updates

if __name__ == "__main__":
    app = SpektrometerApp()
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        app.cleanup()
