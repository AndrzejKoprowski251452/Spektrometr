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

def get_color_gradient(n):
    colors = list(mcolors.TABLEAU_COLORS.values())
    gradient = [colors[i % len(colors)] for i in range(n)]
    return gradient
import numpy as np

config = json.load(open('api.settings.json'))
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
        for w in self.window.winfo_children():
            w.config(bg=self.DGRAY,fg='lightgray',highlightbackground='white')
        
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
        self.geometry(f'{self.winfo_width()}x{self.winfo_height()}')
        self.menu_button = CButton(self.title_bar, text=' ≡ ', command=self.show_menu)
        self.menu_button.pack(side=LEFT,before=self.title_bar_title, ipadx=7,fill=Y)
        
        menu = Menu(self, tearoff=0,background='darkgray',foreground='lightgray',activebackground=self.RGRAY,activeforeground='white',bd=0,borderwidth=0,relief='flat')
        menu.config(bg=self.DGRAY,fg='lightgray',relief='flat',bd=0,borderwidth=0,activebackground=self.RGRAY,activeforeground='white',disabledforeground='gray')
        menu.add_command(label="Options", command=lambda: Options(self))
        self.menu_bar = menu
        
    def show_menu(self):
        self.menu_bar.post(self.menu_button.winfo_rootx(), self.menu_button.winfo_rooty() + self.menu_button.winfo_height())


class CustomToplevel(Toplevel, CustomWindow):
    def __init__(self, *args, **kwargs):
        Toplevel.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.overrideredirect(True)
        self.config(bg=self.DGRAY, highlightthickness=0)
        
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

class HotmapWindow(CustomToplevel):
    def __init__(self, parent, i, n, image):
        CustomToplevel.__init__(self, parent)
        self.set_title(f'{i}')

        self.image = image
        self.z_values = np.linspace(-10, 10, 100)
        self.current_z_index = 0

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        self.slider = Scale(self.window, from_=0, to=len(self.z_values)-1, orient=HORIZONTAL, command=self.update_plot, bg=self.DGRAY, fg='lightgray', troughcolor=self.DGRAY,borderwidth=0,highlightthickness=1,highlightbackground=self.MGRAY,highlightcolor=self.LGRAY)
        self.slider.pack(fill=X, padx=10, pady=10)

        self.fig = plt.figure(figsize=(5, 5), facecolor=self.DGRAY)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.canvas.get_tk_widget().config(bg=self.DGRAY)

    def update_plot(self, val=None):
        self.current_z_index = int(self.slider.get())
        z = self.z_values[self.current_z_index]

        x = np.arange(0, self.image.width, 1)
        y = np.arange(0, self.image.height, 1)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/100+z) * np.cos(Y/100+z) * z

        self.ax.clear()
        self.ax.patch.set_facecolor(self.DGRAY)
        self.ax.xaxis.set_pane_color((0,0,0,0))
        self.ax.yaxis.set_pane_color((0,0,0,0))
        self.ax.plot_surface(X, Y, Z, cmap=cm.hot, norm=Normalize(vmin=np.min(Z), vmax=np.max(Z)))
        self.ax.set_xlabel('Oś X', color='white')
        self.ax.set_ylabel('Oś Y', color='white')
        self.ax.set_zlabel('Oś Z', color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')

        self.canvas.draw()

class Options(CustomToplevel):
    def __init__(self, parent):
        CustomToplevel.__init__(self, parent)
        self.geometry('1200x800')
        self.set_title("Options")

        self.step_x = IntVar(value=100)
        self.step_y = IntVar(value=100)
        self.offset = IntVar(value=0)
        self.square_width = IntVar(value=50)
        self.square_height = IntVar(value=50)

        self.create_options()
        for w in self.winfo_children():
            w.config(bg=self.DGRAY, fg='lightgray', highlightbackground='white')
            for w in w.winfo_children():
                w.config(bg=self.DGRAY, fg='lightgray', highlightbackground='white')

    def create_options(self):
        # Główna ramka dla opcji
        frame = Frame(self.window, bg=self.DGRAY)
        frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        # Nagłówek
        Label(frame, text="Options", font=("Helvetica", 16, "bold"), bg=self.DGRAY, fg='lightgray').grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Step X
        Label(frame, text="Step X:", bg=self.DGRAY, fg='lightgray').grid(row=1, column=0, sticky=W, pady=5)
        self.step_x_entry = Entry(frame, textvariable=self.step_x, bg=self.RGRAY, fg='lightgray', insertbackground='lightgray')
        self.step_x_entry.grid(row=1, column=1, sticky=EW, pady=5)

        # Step Y
        Label(frame, text="Step Y:", bg=self.DGRAY, fg='lightgray').grid(row=2, column=0, sticky=W, pady=5)
        self.step_y_entry = Entry(frame, textvariable=self.step_y, bg=self.RGRAY, fg='lightgray', insertbackground='lightgray')
        self.step_y_entry.grid(row=2, column=1, sticky=EW, pady=5)

        # Offset
        Label(frame, text="Offset:", bg=self.DGRAY, fg='lightgray').grid(row=3, column=0, sticky=W, pady=5)
        self.offset_entry = Entry(frame, textvariable=self.offset, bg=self.RGRAY, fg='lightgray', insertbackground='lightgray')
        self.offset_entry.grid(row=3, column=1, sticky=EW, pady=5)

        # Square Width
        Label(frame, text="Square Width:", bg=self.DGRAY, fg='lightgray').grid(row=4, column=0, sticky=W, pady=5)
        self.square_width_entry = Entry(frame, textvariable=self.square_width, bg=self.RGRAY, fg='lightgray', insertbackground='lightgray')
        self.square_width_entry.grid(row=4, column=1, sticky=EW, pady=5)
        
        Label(frame, text="Square Height:", bg=self.DGRAY, fg='lightgray').grid(row=5, column=0, sticky=W, pady=5)
        self.square_width_entry = Entry(frame, textvariable=self.square_height, bg=self.RGRAY, fg='lightgray', insertbackground='lightgray')
        self.square_width_entry.grid(row=5, column=1, sticky=EW, pady=5)

        # Przyciski
        button_frame = Frame(frame, bg=self.DGRAY)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)

        CButton(button_frame, text="Import Settings", command=self.import_settings).pack(side=LEFT, padx=10)
        CButton(button_frame, text="Apply", command=self.apply_settings).pack(side=LEFT, padx=10)

        # Ustawienia kolumn
        frame.columnconfigure(1, weight=1)

    def import_settings(self):
        global config
        file_path = filedialog.askopenfilename(title="Select Settings File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            config = json.load(open(file_path))
            print(f"Importing settings from {file_path}")
        self.focus()

    def apply_settings(self):
        settings = {
            "step_x": self.step_x.get(),
            "step_y": self.step_y.get(),
            "offset": self.offset.get(),
            "width": self.square_width.get(),
            "height": self.square_height.get(),
        }
        with open('options.json', 'w') as f:
            json.dump(settings, f, indent=4)