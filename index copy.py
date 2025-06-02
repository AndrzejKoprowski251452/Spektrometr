from tkinter import *
from tkinter import ttk,filedialog
from pixelinkWrapper import*
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from ctypes import*
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
from PIL import ImageDraw
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import asyncio
import serial.tools.list_ports
import time
import asyncio
from async_tkinter_loop import async_mainloop
import os
import csv
import glob
import sys

options = json.load(open('options.json'))
def get_color_gradient(n):
    colors = list(mcolors.TABLEAU_COLORS.values())
    gradient = [colors[i % len(colors)] for i in range(n)]
    return gradient
import numpy as np

options = json.load(open('options.json'))
LGRAY = '#232323'
DGRAY = '#161616'
RGRAY = '#2c2c2c'
MGRAY = '#1D1c1c'

class SonyCamera:
    def __init__(self):
        self.camera = Camera()
        self.actions = Actions(self.camera)
        self.camera.startLiveview()

class CustomWindow:
    def __init__(self, *args, **kwargs):
        self.tk_title = "Custom Window"
        self.LGRAY = LGRAY
        self.DGRAY = DGRAY
        self.RGRAY = RGRAY
        self.MGRAY = MGRAY

        self.title_bar = Frame(self, bg=self.RGRAY, relief='raised', bd=0, highlightthickness=1, highlightbackground=self.MGRAY, highlightcolor=self.MGRAY)
        self.close_button = Button(self.title_bar, text='  ×  ', command=self.destroy, bg=self.RGRAY, padx=2, pady=2, font=("calibri", 13), bd=0, fg='lightgray', highlightthickness=0)
        self.minimize_button = Button(self.title_bar, text=' 🗕 ', command=self.minimize_me, bg=self.RGRAY, padx=2, pady=2, bd=0, fg='lightgray', font=("calibri", 13), highlightthickness=0)
        self.title_bar_title = Label(self.title_bar, text=self.tk_title, bg=self.RGRAY, bd=0, fg='lightgray', font=("helvetica", 10), highlightthickness=0)
        self.window = Frame(self, bg=self.DGRAY, highlightthickness=1, highlightbackground=self.MGRAY, highlightcolor=self.MGRAY)

        self.title_bar.pack(fill=X)
        self.title_bar_title.pack(side=LEFT, padx=10)
        self.close_button.pack(side=RIGHT, ipadx=7, ipady=1)
        self.minimize_button.pack(side=RIGHT, ipadx=7, ipady=1)
        self.window.pack(expand=1, fill=BOTH)
        self.window.pack_propagate(1)

        self.title_bar.bind('<Button-1>', self.get_pos)
        self.title_bar_title.bind('<Button-1>', self.get_pos)

        self.close_button.bind('<Enter>', lambda e: self.changex_on_hovering())
        self.close_button.bind('<Leave>', lambda e: self.returnx_to_normalstate())
        self.minimize_button.bind('<Enter>', lambda e: self.change__on_hovering())
        self.minimize_button.bind('<Leave>', lambda e: self.return__to_normalstate())

        self.custom_scroll()

        if self.winfo_class() == 'Tk':
            self.bind("<Expose>", lambda e: self.deminimize())
        self.after(10, lambda: self.set_appwindow())
      
    def update_colors(self):  
        def set_colors(widget):
            if isinstance(widget, (Label, Button, Entry, Text, Frame,LabelFrame, Toplevel, Canvas, Scrollbar)):
                try:
                    widget.options(bg=self.DGRAY, fg='lightgray', highlightbackground='white')
                except Exception:
                    pass
            for child in widget.winfo_children():
                set_colors(child)
        set_colors(self.window)
        
    def get_pos(self,event):
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
        self.title_bar_title.bind('<B1-Motion>', move_window)
        self.title_bar_title.bind('<ButtonRelease-1>', release_window)

    def set_appwindow(self): 
        GWL_EXSTYLE = -20
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080
        hwnd = windll.user32.GetParent(self.winfo_id())
        stylew = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        stylew = stylew & ~WS_EX_TOOLWINDOW
        stylew = stylew | WS_EX_APPWINDOW
        res = windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, stylew)
    
        self.wm_withdraw()
        self.after(10, lambda: self.wm_deiconify())
        self.update()
        self.geometry(f'{self.winfo_width()}x{self.winfo_height()}')
        
    def minimize_me(self):
        self.overrideredirect(False)
        self.attributes('-alpha',0)
        self.wm_state('iconic')
        
    def deminimize(self):
        self.overrideredirect(True)
        self.attributes('-alpha',1)
        self.wm_state('zoomed')

    def changex_on_hovering(self):
        self.close_button['bg']='red'
        
    def returnx_to_normalstate(self):
        self.close_button['bg']=self.RGRAY
        
    def change__on_hovering(self):
        self.minimize_button['bg']='gray'
        
    def return__to_normalstate(self):
        self.minimize_button['bg']=self.RGRAY
        
    def set_title(self, title):
        self.tk_title = title
        self.title_bar_title.config(text=self.tk_title)
        self.title(self.tk_title)
        
    def custom_scroll(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Horizontal.TScrollbar",
            gripcount=0,
            background=self.LGRAY,
            darkcolor=self.DGRAY,
            lightcolor=self.DGRAY,
            troughcolor=self.DGRAY,
            bordercolor=self.DGRAY,
            arrowcolor=self.DGRAY,
            activebackground=self.DGRAY,
            deactivebackground=self.DGRAY,
        )
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=self.LGRAY,
            darkcolor=self.DGRAY,
            lightcolor=self.DGRAY,
            troughcolor=self.DGRAY,
            bordercolor=self.DGRAY,
            arrowcolor=self.DGRAY,
            activebackground=self.DGRAY,
            deactivebackground=self.DGRAY,
        )

class CButton(Button):
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self.config(
            bg=RGRAY,
            padx=2,
            pady=2,
            bd=0,
            fg='lightgray',
            highlightthickness=0,
            relief='flat'
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<ButtonPress-1>', self.on_press)
        self.bind('<ButtonRelease-1>', self.on_release)

    def on_enter(self, event,color='gray'):
        self.config(bg=color)

    def on_leave(self, event):
        self.config(bg=RGRAY)

    def on_press(self, event):
        self.config(relief='sunken')

    def on_release(self, event):
        self.config(relief='flat')

class CustomTk(Tk, CustomWindow):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.tk_title = "Arczy Puszka"
        self.overrideredirect(True)
        self.config(bg=self.DGRAY, highlightthickness=0)
        self.state('zoomed')
            
class CustomToplevel(Toplevel, CustomWindow):
    def __init__(self, *args, **kwargs):
        Toplevel.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.overrideredirect(True)
        self.window = self
        self.config(bg=self.DGRAY, highlightthickness=0)
        self.iconbitmap("icon.ico")
        
class StreamToFunction:
    def __init__(self, func):
        self.func = func

    def write(self, message):
        if message.strip():
            self.func(message)
    def flush(self):
        pass

class MotionDetector:
    def __init__(self, video_source=1):
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.prev_frame = None

    def detect_movement_direction(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame,1)
        if not ret:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return None, frame

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_frame = gray

        mean_flow = np.mean(flow, axis=(0, 1))
        magnitude = np.linalg.norm(mean_flow)

        if magnitude < 0.1:
            return "No significant movement", frame
        if abs(mean_flow[0]) > abs(mean_flow[1]):
            direction = 'Right' if mean_flow[0] > 0 else 'Left'
        else:
            direction = 'Down' if mean_flow[1] > 0 else 'Up'
        
        return direction, frame

    def release(self):
        self.cap.release()

class HeatMapWindow(CustomToplevel):
    def __init__(self, parent, i, n, image):
        CustomToplevel.__init__(self, parent)
        self.set_title(f'Pomiar {i}')
        self.geometry('1200x800')

        self.image = image
        n = np.array(n, dtype=object)
        xs = sorted(set([row[0] for row in n]))
        ys = sorted(set([row[1] for row in n]))
        nx, ny = len(xs), len(ys)
        self.xmin = parent.xmin_var.get()
        self.xmax = parent.xmax_var.get()
        spectrum_len = len(n[0][2])
        self.cube = np.zeros((nx, ny, spectrum_len))
        for row in n:
            ix = xs.index(row[0])
            iy = ys.index(row[1])
            self.cube[ix, iy, :] = row[2]
        self.current_lambda = 0

        self.lambdas = np.linspace(int(self.xmin), int(self.xmax), spectrum_len)

        self.create_widgets()
        self.update_heatmap()
        self.update_profile()

    def create_widgets(self):
        self.slider = Scale(
            self.window,
            from_=0,
            to=self.cube.shape[2] - 1,
            orient=HORIZONTAL,
            command=self.on_slider,
            bg=self.DGRAY,
            fg='lightgray',
            troughcolor=self.DGRAY,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.MGRAY,
            highlightcolor=self.LGRAY,
            length=600
        )
        self.slider.pack(fill=X, padx=10, pady=10)

        self.fig = plt.figure(figsize=(8, 8), facecolor=self.DGRAY)
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.canvas.get_tk_widget().config(bg=self.DGRAY)

    def on_slider(self, val):
        self.current_lambda = int(val)
        self.update_heatmap()
        self.update_profile()

    def update_heatmap(self):
        self.ax.clear()
        data = self.cube[:, :, self.current_lambda]
        X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
        self.ax.plot_surface(X, Y, data.T, cmap='hot')
        self.ax.patch.set_facecolor(self.DGRAY)
        lambda_val = self.lambdas[self.current_lambda]
        self.ax.set_title(f"Heatmapa 3D dla długości fali {lambda_val:.0f}", color='white')
        self.ax.set_xlabel("X", color='white')
        self.ax.set_ylabel("Y", color='white')
        self.ax.set_zlabel("Jasność", color='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_profile(self):
        self.ax2.clear()
        mean_profile = self.cube.mean(axis=(0,1))
        self.ax2.plot(self.lambdas, mean_profile, color='orange', label="Spektrum (średnia)")
        lambda_val = self.lambdas[self.current_lambda]
        self.ax2.axvline(lambda_val, color='red', linestyle='--', label=f"λ={lambda_val:.0f}")
        self.ax2.set_title("Profil modów (pełne spektrum)", color='white')
        self.ax2.set_xlabel("Długość fali (pixel lub nm)", color='white')
        self.ax2.set_ylabel("Jasność", color='white')
        self.ax2.tick_params(axis='x', colors='white')
        self.ax2.tick_params(axis='y', colors='white')
        self.ax2.set_facecolor(self.DGRAY)
        self.ax2.legend(facecolor=self.DGRAY, edgecolor='gray', labelcolor='white')
        self.fig.tight_layout()
        self.canvas.draw_idle()

class App(CustomTk):
    def __init__(self):
        super().__init__()
        self.title("Arczy Puszka")
        self.iconbitmap("icon.ico")
        self.geometry('1200x800')
        
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Custom.TNotebook', background=self.DGRAY, borderwidth=0)
        style.configure('Custom.TNotebook.Tab', background=self.DGRAY, foreground='white', font=('Segoe UI', 11))
        style.map('Custom.TNotebook.Tab', background=[('selected', self.DGRAY)])
        style.layout("Custom.TNotebook.Tab",
    [('Notebook.tab', {'sticky': 'nswe', 'children':
        [('Notebook.padding', {'side': 'top', 'sticky': 'nswe', 'children':
            [('Notebook.label', {'side': 'top', 'sticky': ''})],
        })],
    })]
)
        self.notebook = ttk.Notebook(self.window, style="Custom.TNotebook")
        self.notebook.pack(fill=BOTH, expand=True)
        
        self.options_window = None
        
        sys.stdout = StreamToFunction(self.console_data)
        
        self.step_x = IntVar(value=options['step_x'])
        self.step_y = IntVar(value=options['step_y'])
        self.offset = IntVar(value=options['offset'])
        self.square_width = IntVar(value=options['width'])
        self.square_height = IntVar(value=options['height'])

        self.c1 = Frame(self.notebook, bg=self.DGRAY)
        self.c2 = Frame(self.notebook, bg=self.DGRAY)
        self.c3 = Frame(self.notebook, bg=self.DGRAY)
        self.c4 = Frame(self.notebook, bg=self.DGRAY)
        self.c5 = Frame(self.notebook, bg=self.DGRAY)
        self.c6 = Frame(self.notebook, bg=self.DGRAY)

        self.notebook.add(self.c1, text="Camera")
        self.notebook.add(self.c2, text="Controls")
        self.notebook.add(self.c3, text="Spectrometr View")
        self.notebook.add(self.c4, text="Spectrum")
        self.notebook.add(self.c5, text="Results")
        self.notebook.add(self.c6, text="Settings")

        self.tasks_frame = Frame(self.c2, bg=self.DGRAY)
        self.tasks_frame.pack(fill=X, padx=10, pady=(0, 10))
        
        self.console = Text(self.c2, background=self.DGRAY, fg=self.LGRAY, foreground='white')
        self.console.pack(fill=BOTH, expand=True, padx=10, pady=(10, 5))

        self.canvas = Canvas(self.c5, bg=self.DGRAY, bd=0, highlightthickness=0)
        self.scrollbar_x = ttk.Scrollbar(self.c5, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y = ttk.Scrollbar(self.c5, orient="vertical", command=self.canvas.yview)
        self.button_frame = Frame(self.canvas, bg=self.DGRAY)

        self.frame_label = Label(self.c1, bd=0, highlightthickness=0)
        self.spectrometr_canvas = Canvas(self.c3, bg=self.DGRAY, bd=0, highlightthickness=0)
        self.spectrometr_canvas.grid(row=0, column=0, sticky="nsew")
        self.c3.grid_rowconfigure(0, weight=1)
        self.c3.grid_columnconfigure(0, weight=1)

        self.original_image = Image.open("3.bmp")
        self.scale = 1.0
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.spectrometr_image = self.spectrometr_canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

        self.spectrometr_canvas.bind("<MouseWheel>", self.zoom)
        self.spectrometr_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.spectrometr_canvas.bind("<B1-Motion>", self.pan)
        
        self.ports = []
        self.connected = False
        self.update()
        if len(list(serial.tools.list_ports.comports())) != 0 and any(['COM' in p[0] for p in serial.tools.list_ports.comports()]):
            self.connected = True
            for i in serial.tools.list_ports.comports():
                print(i[0])
                if i[0] == 'COM5' or i[0] == 'COM9':
                    s = serial.Serial(i[0])
                    self.ports.append(s)
        else:
            self.connected = False

        self.left = CButton(self.c1, text='←', width=2, height=1, command=lambda: self.move('l'))
        self.right = CButton(self.c1, text='→', width=2, height=1, command=lambda: self.move('r'))
        self.up = CButton(self.c1, text='↑', width=2, height=1, command=lambda: self.move('u'))
        self.down = CButton(self.c1, text='↓', width=2, height=1, command=lambda: self.move('d'))
        self.origin = CButton(self.c1, text='o', width=2, height=1, command=lambda: self.move('o'))
        self.v = Label(self.c1, text='x:0,y:0', background=self.DGRAY, fg='white', anchor='center')
        self.s = CButton(self.c1, text='s', width=2, height=1, command=self.start_sequence)

        self.left.place(x=25, y=50)
        self.right.place(x=75, y=50)
        self.up.place(x=50, y=25)
        self.down.place(x=50, y=75)
        self.origin.place(x=50, y=50)
        self.s.place(x=100, y=50)
        self.v.place(x=42, y=100)
        self.frame_label.place(x=0, y=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set, background=self.DGRAY)
        self.canvas.bind("<Configure>", lambda event: self.update_canvas())
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")
        self.c5.grid_rowconfigure(0, weight=1)
        self.c5.grid_columnconfigure(0, weight=1)
        
        self.cameraIndex=1
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = []
        
        self.tasks = [
            {"name": "Camera 1", "status": StringVar(value="Pending"),"event": None},
            {"name": "Camera 2", "status": StringVar(value="Pending"),"event": None},
            {"name": "Auto exposure", "status": StringVar(value="Not calibrated"),"event": None},
            {"name": "Auto white balance", "status": StringVar(value="Not calibrated"),"event": None},
        ]
        
        self.image = None  
        self.connected = False
        self.calibrated = False
            
        self.update_colors()
        self.create_options()
        
        self.spectrometr_canvas.bind("<MouseWheel>", self.zoom)
        self.spectrometr_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.spectrometr_canvas.bind("<B1-Motion>", self.pan)
        
        self.xmin_var = StringVar(value=options['xmin'])
        self.xmax_var = StringVar(value=options['xmax'])

        Label(self.c4, text="X min:", bg=self.DGRAY, fg='lightgray').pack(side=LEFT, padx=5)
        self.xmin_entry = Entry(self.c4, textvariable=self.xmin_var, width=8)
        self.xmin_entry.pack(side=LEFT, padx=5)
        Label(self.c4, text="X max:", bg=self.DGRAY, fg='lightgray').pack(side=LEFT, padx=5)
        self.xmax_entry = Entry(self.c4, textvariable=self.xmax_var, width=8)
        self.xmax_entry.pack(side=LEFT, padx=5)
        CButton(self.c4, text="Ustaw zakres X", command=self.set_x_range).pack(side=LEFT, padx=10)
        
        self.hCamera = None
        
        self.draw_measurements()
        
    def create_options(self):
        frame = Frame(self.c6, bg=self.DGRAY)
        frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        Label(frame, text="Options", font=("Helvetica", 16, "bold"), bg=self.DGRAY, fg='white').grid(row=0, column=0, columnspan=2, pady=(0, 20))

        Label(frame, text="Step X:", bg=self.DGRAY, fg='white').grid(row=1, column=0, sticky=W, pady=5)
        self.step_x_entry = Entry(frame, textvariable=self.step_x, bg=self.RGRAY, fg='white', insertbackground='white')
        self.step_x_entry.grid(row=1, column=1, sticky=EW, pady=5)

        Label(frame, text="Step Y:", bg=self.DGRAY, fg='white').grid(row=2, column=0, sticky=W, pady=5)
        self.step_y_entry = Entry(frame, textvariable=self.step_y, bg=self.RGRAY, fg='white', insertbackground='white')
        self.step_y_entry.grid(row=2, column=1, sticky=EW, pady=5)

        Label(frame, text="Offset:", bg=self.DGRAY, fg='white').grid(row=3, column=0, sticky=W, pady=5)
        self.offset_entry = Entry(frame, textvariable=self.offset, bg=self.RGRAY, fg='white', insertbackground='white')
        self.offset_entry.grid(row=3, column=1, sticky=EW, pady=5)

        Label(frame, text="Square Width:", bg=self.DGRAY, fg='white').grid(row=4, column=0, sticky=W, pady=5)
        self.square_width_entry = Entry(frame, textvariable=self.square_width, bg=self.RGRAY, fg='white', insertbackground='white')
        self.square_width_entry.grid(row=4, column=1, sticky=EW, pady=5)

        Label(frame, text="Square Height:", bg=self.DGRAY, fg='white').grid(row=5, column=0, sticky=W, pady=5)
        self.square_height_entry = Entry(frame, textvariable=self.square_height, bg=self.RGRAY, fg='white', insertbackground='white')
        self.square_height_entry.grid(row=5, column=1, sticky=EW, pady=5)
        
        ports = list(serial.tools.list_ports.comports())
        port_choices = [f"{p.device} - {p.description}" for p in ports]
        port_values = [p.device for p in ports]

        Label(frame, text="Port X:", bg=self.DGRAY, fg='white').grid(row=6, column=0, sticky=W, pady=5)
        self.port_x_var = StringVar(value=options.get('port_x', 'COM5'))
        self.port_x_combo = ttk.Combobox(frame, textvariable=self.port_x_var, values=port_choices, state="readonly")
        for i, p in enumerate(ports):
            if p.device == options.get('port_x'):
                self.port_x_combo.current(i)
                break
        self.port_x_combo.grid(row=6, column=1, sticky=EW, pady=5)

        Label(frame, text="Port Y:", bg=self.DGRAY, fg='white').grid(row=7, column=0, sticky=W, pady=5)
        self.port_y_var = StringVar(value=options.get('port_y', 'COM9'))
        self.port_y_combo = ttk.Combobox(frame, textvariable=self.port_y_var, values=port_choices, state="readonly")
        for i, p in enumerate(ports):
            if p.device == options.get('port_y'):
                self.port_y_combo.current(i)
                break
        self.port_y_combo.grid(row=7, column=1, sticky=EW, pady=5)

        button_frame = Frame(frame, bg=self.DGRAY)
        button_frame.grid(row=8, column=0, columnspan=2, pady=20)

        CButton(button_frame, text="Import Settings", command=self.import_settings).pack(side=LEFT, padx=10)
        CButton(button_frame, text="Apply", command=self.apply_settings).pack(side=LEFT, padx=10)

        frame.columnconfigure(1, weight=1)

    def import_settings(self):
        global options
        file_path = filedialog.askopenfilename(title="Select Settings File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            options = json.load(open(file_path))
            print(f"Importing settings from {file_path}")
        self.focus()

    def apply_settings(self):
        port_x = self.port_x_combo.get().split(" - ")[0]
        port_y = self.port_y_combo.get().split(" - ")[0]
        settings = {
            "step_x": self.step_x.get(),
            "step_y": self.step_y.get(),
            "offset": self.offset.get(),
            "width": self.square_width.get(),
            "height": self.square_height.get(),
            "await":0.01,
            "xmin": self.xmin_var.get(),
            "xmax": self.xmax_var.get(),
            "port_x": port_x,
            "port_y": port_y,
        }
        with open('options.json', 'w') as f:
            json.dump(settings, f, indent=4)

    def zoom(self, event):
        old_scale = self.scale
        if event.delta > 0 and self.scale < 4:
            self.scale += 0.1
        elif event.delta < 0 and self.scale > 0.5:
            self.scale -= 0.1
        if self.scale != old_scale:
            canvas = self.spectrometr_canvas
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            resized_image = self.original_image.resize(
                (int(self.original_image.width * self.scale), int(self.original_image.height * self.scale))
            )
            self.image_tk = ImageTk.PhotoImage(resized_image)
            canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
            bbox = canvas.bbox(self.spectrometr_image)
            if bbox:
                img_x = x * self.scale / old_scale
                img_y = y * self.scale / old_scale
                canvas.xview_moveto((img_x - event.x) / (resized_image.width - canvas.winfo_width()))
                canvas.yview_moveto((img_y - event.y) / (resized_image.height - canvas.winfo_height()))

    def start_pan(self, event):
        self.spectrometr_canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.spectrometr_canvas.scan_dragto(event.x, event.y, gain=1)
        
    def cameraI(self,i):
        self.cameraIndex=i
        if not self.calibrated:
            self.detector = MotionDetector(self.cameraIndex)
        else:
            self.detector = cv2.VideoCapture(self.cameraIndex) 
        
    def console_data(self,f):
        readable_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        self.console.insert(INSERT, f'{readable_time}: {f}\n')
        self.console.see("end")

    def create_task_list(self):
        for i, task in enumerate(self.tasks):
            frame = Frame(self.tasks_frame, bg=self.DGRAY)
            frame.pack(fill=X, pady=5)

            label = Label(frame, text=task["name"], bg=self.DGRAY, fg='lightgray')
            label.grid(row=0, column=0, padx=10)

            status_label = Label(frame, textvariable=task["status"], bg=self.DGRAY, fg='red')
            status_label.grid(row=0, column=1, padx=10)

            complete_button = CButton(
                frame,
                text="Complete",
                command=lambda t=task, sl=status_label: self.change_state(t, sl)
            )
            complete_button.grid(row=0, column=2, padx=10)

            options_button = CButton(
                frame,
                text="Options",
                command=lambda t=task: self.show_task_options(t)
            )
            options_button.grid(row=0, column=3, padx=10)

    def change_state(self, task,status_label):
        try:
            status_label.config(fg='green')
            task["status"].set("Completed")
            print(f"Task '{task['name']}' completed successfully.")
        except Exception as e:
            status_label.config(fg='red')
            task["status"].set("Failed")
            print(f"Error completing task '{task['name']}': {e}")

    def show_task_options(self, task):
        if task['name'] == "Camera 1":
            c1=CustomToplevel(self)
            c1.set_title("Camera 1 settings")
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        Button(c1.window,text=f'Camera {i}',command=lambda i=i:self.cameraI(i)).pack(pady=5)
                        cap.release()
                    else:
                        break
                except:
                    break
        elif task['name'] == "Camera 2":
            ret = PxLApi.getNumberCameras()
            c1=CustomToplevel(self)
            c1.set_title("Camera 2 settings")
            if PxLApi.apiSuccess(ret[0]):
                cameras = ret[1]
                print ("Found %d Cameras:" % len(cameras))
                for i in range(len(cameras)):
                    Button(c1.window,text="  Serial number - %d" % cameras[i].CameraSerialNum,command=lambda i=i:self.cameraI(i)).pack(pady=5)
            if len(cameras) == 0:
                Label(c1.window,text="No cameras found", bg=self.DGRAY, fg='lightgray').pack(pady=5)
        elif task['name'] == "Etalonu calibration":
            pass
        elif task['name'] == "Grid calibration":
            pass
        elif task['name'] == "Auto exposure":
            if self.hCamera is not None:
                exposure = 0
                params = [exposure]
                ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.EXPOSURE)
                params = ret[2]
                flags = PxLApi.FeatureFlags.AUTO
                ret = PxLApi.setFeature(self.hCamera, PxLApi.FeatureId.EXPOSURE, flags, params)
        elif task['name'] == "Auto white balance":
            if self.hCamera is not None:
                ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.WHITE_SHADING)
                assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
        print(f"Options for {task['name']}")
        
    # def find_mods(self):
    #     img = np.array(self.original_image.convert('L'))
    #     bright_mask = img > 100
    #     coords = np.column_stack(np.where(bright_mask))
    #     if coords.size == 0:
    #         self.y = []
    #         return

    #     y_min, x_min = coords.min(axis=0)
    #     y_max, x_max = coords.max(axis=0)

    #     roi = img[y_min:y_max+1, x_min:x_max+1]
    #     _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     self.y = []
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         x += x_min
    #         y += y_min
    #         mask = np.zeros_like(img)
    #         cv2.drawContours(mask, [contour], -1, 255, -1)
    #         mean_val = 1/(cv2.mean(img, mask=mask)[0]/(w*h))

    #         cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #         #cv2.line(img_color, (x + w // 2, 0), (x + w // 2, img_color.shape[0]), (0, 255, 0), 2)
    #         cv2.putText(img_color, f'{mean_val:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #         self.y.append((x + w // 2, mean_val))

    #     self.original_image = Image.fromarray(img_color)
    #     self.image_tk = ImageTk.PhotoImage(self.original_image)
    #     self.spectrometr_canvas.itemoptions(self.spectrometr_image, image=self.image_tk)

    # def draw_lines_on_image(self):
    #     if self.image_tk is None:
    #         image_array = np.array(self.original_image)
            
    #         if len(image_array.shape) == 3 and image_array.shape[2] == 4:
    #             gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
    #         elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
    #             gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    #         else:
    #             gray_image = image_array
    #         gray_image = cv2.GaussianBlur(gray_image, (13, 13), 0)
    #         gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #         grad = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=5)
    #         grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), grad)
    #         line_image = np.zeros_like(grad)
    #         contours, _ = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         line_positions = []
    #         for contour in contours:
    #             if len(contour) > 0:
    #                 rect = cv2.minAreaRect(contour)
    #                 box = cv2.boxPoints(rect)
    #                 box = np.int32(box)
    #                 x1 = (box[0][0] + box[3][0]) // 2
    #                 y1 = (box[0][1] + box[3][1]) // 2
    #                 x2 = (box[1][0] + box[2][0]) // 2
    #                 y2 = (box[1][1] + box[2][1]) // 2
    #                 cv2.line(line_image, (x1, y1), (x2, y2), (255), 1)
    #                 line_positions.append((x1, y1, x2, y2))
    #         img = Image.fromarray(line_image)
    #         img = img.resize((int(self.original_image.width * self.scale), int(self.original_image.height * self.scale)))
    #         self.image_tk = ImageTk.PhotoImage(img)
    #         self.original_image = img
    #         self.spectrometr_canvas.itemoptions(self.spectrometr_image, image=self.image_tk)
    #         return line_positions
    #     return []

    def create_spectrum_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 2), facecolor=self.DGRAY)
        self.ax.set_facecolor(self.DGRAY)
        self.x = np.linspace(0, 2048, 2048)
        self.y = np.zeros(2048)
        (self.spectrum_line,) = self.ax.plot(self.x, self.y, color='darkgreen')
        self.ax.grid()
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig, master=self.c4)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill=BOTH, expand=True, pady=(10, 0), padx=10)
        self.spectrum_canvas.mpl_connect('scroll_event', self.spectrum_zoom)
        self.spectrum_canvas.mpl_connect('motion_notify_event', self.spectrum_pan)
        self.spectrum_canvas.mpl_connect('button_press_event', self.spectrum_pan_start)
        self.spectrum_canvas.mpl_connect('button_release_event', self.spectrum_pan_stop)
        self._pan_start = None

    def update_spectrum_plot(self):
        img = np.array(self.original_image.convert('L'))
        self.y = img.sum(axis=0)/img.shape[0]
        self.spectrum_line.set_ydata(self.y)
        y_max = max(self.y)
        self.ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1)

        if hasattr(self, "_spectrum_vlines"):
            for l in self._spectrum_vlines:
                try:
                    l.remove()
                except Exception:
                    pass

        xmin = float(self.xmin_var.get())
        xmax = float(self.xmax_var.get())
        vline1 = self.ax.axvline(xmin, color='red', linestyle='--')
        vline2 = self.ax.axvline(xmax, color='red', linestyle='--')
        self._spectrum_vlines = [vline1, vline2]

        self.spectrum_canvas.draw()
    
    def spectrum_zoom(self, event):
        ax = self.ax
        x_min, x_max = ax.get_xlim()
        zoom_factor = 0.8 if event.button == 'up' else 1.25
        xdata = event.xdata
        if xdata is None:
            return
        new_xmin = xdata - (xdata - x_min) * zoom_factor
        new_xmax = xdata + (x_max - xdata) * zoom_factor
        ax.set_xlim(new_xmin, new_xmax)
        self.update_spectrum_plot()

    def spectrum_pan_start(self, event):
        if event.button == 1 and event.xdata is not None:
            self._pan_start = event.xdata

    def spectrum_pan(self, event):
        if self._pan_start is not None and event.button == 1 and event.xdata is not None:
            dx = self._pan_start - event.xdata
            minx, maxx = self.ax.get_xlim()
            self.ax.set_xlim(minx + dx, maxx + dx)
            if hasattr(self, "_spectrum_vlines"):
                for l in self._spectrum_vlines:
                    try:
                        l.remove()
                    except Exception:
                        pass
            self.spectrum_canvas.draw()
            self._pan_start = event.xdata
            self.update_spectrum_plot()
        elif event.button != 1:
            self._pan_start = None

    def spectrum_pan_stop(self, event):
        self._pan_start = None
        self.update_spectrum_plot()

    async def update_video_feed(self):
        while True:
            try:
                if not self.winfo_exists():
                    break
                if self.detector and not self.calibrated:
                    direction, frame = self.detector.detect_movement_direction()
                    if direction:
                        self.direction.set(f"Movement: {direction}")
                        if self.connected:
                            await asyncio.sleep(0.1)
                            self.move('r')
                            if direction == 'left' or direction == 'right':
                                lh = self.ports[0]
                                lv = self.ports[1]
                            else:
                                lv = self.ports[1]
                                lh = self.ports[0]
                            self.ports[0] = lh
                            self.ports[1] = lv
                            self.calibrated = True
                if self.calibrated:
                    self.detector = cv2.VideoCapture(self.cameraIndex)
                else:
                    _, frame = self.detector.detect_movement_direction()
                    #frame = cv2.flip(frame, 1)
                    if frame is not None:
                        height, width, _ = frame.shape
                        center_x, center_y = width // 2, height // 2
                        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150,150), 1)
                        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150,150), 1)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                        rgb_frame = cv2.resize(rgb_frame, (int(self.winfo_width()), int(self.winfo_height())))
                        self.image = Image.fromarray(rgb_frame)
                        image_tk = ImageTk.PhotoImage(image=self.image)
                        self.frame_label.configure(image=image_tk)
                        self.frame_label.image = image_tk
                        self.update_spectrum_plot()
            except TclError:
                break 
            await asyncio.sleep(0.01)
            
    def update_canvas(self):
        self.button_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    async def make_sequence(self):
        options = json.load(open('options.json'))
        if self.connected:
            import time
            start = time.time()
            spectra = []
            self.move('l', options['width']/2)
            self.move('d', options['height']/2)
            await asyncio.sleep(options["await"])
            xmin = 0
            xmax = options['width']
            ymin = 0
            ymax = options['height']
            frame_count = 0
            for i in range(xmin, xmax, options['step_x']):
                self.move('r', options['step_x'])
                await asyncio.sleep(options["await"])
                for j in range(ymin, ymax, options['step_y']):
                    self.move('u', options['step_y'])
                    await asyncio.sleep(options["await"])
                    x_min = int(self.xmin_var.get())
                    x_max = int(self.xmax_var.get())
                    spectrum = self.y[x_min:x_max].tolist()
                    spectra.append([i, j] + spectrum)
                    frame_count += 1
                    print(f"Frame {frame_count}: x={i}, y={j}")
                self.move('d', options['height'])
            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print("Sequence completed! : {} frames in {:.2f} seconds, FPS: {:.2f}".format(frame_count, elapsed, fps))
            print(f"Zebrano {len(spectra)} pomiarów.")
            folder = "pomiar_dane"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"pomiar_{time.strftime('%Y%m%d_%H%M%S')}_spectra.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                for row in spectra:
                    writer.writerow(row)
            print(f"Dane zapisane do: {filename}")
            self.load_measurements()
            spectra.clear()

    def start_sequence(self):
        asyncio.create_task(self.make_sequence())
        
    def move(self, dir, step=0):
        if step == 0:
            step_x = options['step_x']
            step_y = options['step_y']
        else:
            step_x = step_y = step
        if self.connected:
            if dir == 'r':
                self.ports[0].write((f"M:1+P{step_x}\r\n").encode())
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'l':
                self.ports[0].write((f"M:1-P{step_x}\r\n").encode())
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'u':
                self.ports[1].write((f"M:1+P{step_y}\r\n").encode())
                self.ports[1].write('G:\r\n'.encode())
            elif dir == 'd':
                self.ports[1].write((f"M:1-P{step_y}\r\n").encode())
                self.ports[1].write('G:\r\n'.encode())
            elif dir == 'o':
                self.ports[0].write((f"H:1\r\n").encode())
                self.ports[1].write((f"H:1\r\n").encode())

    def async_loop(self):
        self.create_spectrum_plot()
        self.start_camera()
        self.start_spectrometr()
        asyncio.create_task(self.update_video_feed())
        self.create_task_list()
        self.load_measurements()
        
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
        
        Label(self.c1, text="Detected Movement Direction:",background=self.DGRAY,fg='lightgray').grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction,background=self.DGRAY,fg='lightgray')
        self.direction_label.grid(row=0,column=1)
        
        self.set_x_range()

    def start_camera(self):
        if not self.calibrated:
            self.detector = MotionDetector(self.cameraIndex)
        else:
            self.detector = None
        
    def start_spectrometr(self):
        ret = PxLApi.initialize(0)
        if not(PxLApi.apiSuccess(ret[0])):
            print("Error: Unable to initialize a camera! rc = %i" % ret[0])
            self.hCamera = None
            return 1
        else:
            self.hCamera = ret[1]
        ret = PxLApi.initialize(0, PxLApi.InitializeExFlags.ISSUE_STREAM_STOP)
        if PxLApi.apiSuccess(ret[0]) and self.hCamera is not None:
            ret = PxLApi.loadSettings(self.hCamera, PxLApi.Settings.SETTINGS_FACTORY)
            if not PxLApi.apiSuccess(ret[0]):
                print("Could not load factory settings!")
            else:
                print("Factory default settings restored")
        if self.hCamera is not None:
            ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)

    def draw_measurements(self):
        for i,n in enumerate(self.measurements):
            CButton(self.button_frame,text=f'{i}',command=lambda i=i, n=n: HeatMapWindow(self,i,n,self.image),width=8,height=3).grid(row=i // 40,column=i%40)
        self.update_canvas()

    def set_x_range(self):
        try:
            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            self.ax.set_xlim(xmin, xmax)
            self.spectrum_canvas.draw()
        except Exception as e:
            print(f"Błąd ustawiania zakresu X: {e}")

    def load_measurements(self):
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
         
if __name__ == "__main__":
    app = App()
    app.after_idle(app.async_loop)
    async_mainloop(app)
    if app.hCamera is not None:
        if PxLApi.StreamState == PxLApi.StreamState.START:
            PxLApi.setStreamState(app.hCamera, PxLApi.StreamState.STOP)
            PxLApi.uninitialize(app.hCamera)
