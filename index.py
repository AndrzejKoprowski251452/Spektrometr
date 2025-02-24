from tkinter import *
from tkinter import ttk,filedialog
from pixelinkWrapper import*
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from ctypes import*
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
import time
import json
from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
from PIL import ImageDraw
import matplotlib.colors as mcolors

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
        self.menu_button.pack(side=LEFT,before=self.title_bar_title, ipadx=7, ipady=1)
        
        menu = Menu(self, tearoff=0)
        menu.config(bg=self.DGRAY,fg='lightgray',relief='flat',bd=0,borderwidth=0,activebackground=self.RGRAY,activeforeground='white')
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

class App(CustomTk):
    def __init__(self, *args, **kwargs):
        CustomTk.__init__(self, *args, **kwargs)
        self.set_title("Arczy Puszka")
        self.iconbitmap("icon.ico")
        
        #sys.stdout = StreamToFunction(self.console_data)

        self.c1 = LabelFrame(self.window,text='Camera')
        self.c2 = LabelFrame(self.window,text='Controls')
        self.c3 = LabelFrame(self.window,text='Spectrometr View')
        self.c4 = LabelFrame(self.window,text='Specterum')
        self.c5 = LabelFrame(self.window,text='Analyzed lasers')
        
        self.console = Text(self.c2,background=self.DGRAY,fg='lightgray',height=10,foreground='white')
        self.console.grid(row=10,column=0,sticky='ews')
        
        self.canvas = Canvas(self.c5,bg=self.DGRAY,bd=0,highlightthickness=0)
        self.scrollbar_x = ttk.Scrollbar(self.c5, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y = ttk.Scrollbar(self.c5, orient="vertical", command=self.canvas.yview)
        self.button_frame = Frame(self.canvas,bg=self.DGRAY)
        
        self.frame_label = Label(self.c1,bd=0,highlightthickness=0)
        self.spectrometr_canvas = Canvas(self.c3, bg=self.DGRAY,bd=0,highlightthickness=0)
        self.spectrometr_canvas.pack(expand=True, fill=BOTH)
        
        self.original_image = Image.open("Image.bmp")
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.spectrometr_image = self.spectrometr_canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
        self.scale = 1.0
        
        self.left = CButton(self.c1,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.right = CButton(self.c1,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.up = CButton(self.c1,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.down = CButton(self.c1,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.origin = CButton(self.c1,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.v = Label(self.c1,text='x:0,y:0',background=self.DGRAY,fg='lightgray',anchor='center')
        
        self.c1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.c2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.c3.grid(row=1, column=0, sticky="nsew", padx=5, pady=5, rowspan=2)
        self.c4.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.c5.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.left.place(x=25,y=50)
        self.right.place(x=75,y=50)
        self.up.place(x=50,y=25)
        self.down.place(x=50,y=75)
        self.origin.place(x=50,y=50)
        self.v.place(x=42,y=100)
        self.frame_label.place(x=0,y=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set,background=self.DGRAY)
        self.canvas.bind("<Configure>", lambda event: self.update_canvas())
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")
        self.c5.grid_rowconfigure(0, weight=1)
        self.c5.grid_columnconfigure(0, weight=1)
        
        self.cameraIndex=1
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = [1,2,3]
        self.data = range(100)
        self.speedX = 1
        self.speedY = 1
        
        self.tasks = [
            {"name": "Camera 1", "status": StringVar(value="Pending")},
            {"name": "Camera 2", "status": StringVar(value="Pending")},
            {"name": "Etalonu calibration", "status": StringVar(value="Not calibrated")},
            {"name": "Grid calibration", "status": StringVar(value="Not calibrated")},
            {"name": "Auto exposure", "status": StringVar(value="Not calibrated")},
            {"name": "Auto white balance", "status": StringVar(value="Not calibrated")}
        ]
        
        self.image = None  
        self.spec = None      
        self.connected = False
        self.calibrated = False
        
        self.ports = []
        if len(list(serial.tools.list_ports.comports())) != 0 and 'COM' in list(serial.tools.list_ports.comports()):
            self.connected = True
            for i in list(serial.tools.list_ports.comports()):
                s = serial.Serial(str(i)[:4])
                s.baudrate=9600
                s.BYTESIZES=serial.EIGHTBITS
                s.PARITIES=serial.PARITY_NONE
                s.STOPBITS=serial.STOPBITS_ONE
                s.timeout=1
                s.rtscts=True
                self.ports.append(s)
            self.ports[0].write('list-sensors'.encode())
            print(self.ports[0].readlines())
        else:
            self.connected = False
        
        self.update_colors()
        self.create_widgets()
        self.draw_measurements()
        self.update_sizes()
        
        self.spectrometr_canvas.bind("<MouseWheel>", self.zoom)
        self.spectrometr_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.spectrometr_canvas.bind("<B1-Motion>", self.pan)
        
    def zoom(self, event):
        if event.delta > 0 and self.scale < 4:
            self.scale += 0.1
        elif event.delta < 0 and self.scale > 0.5:
            self.scale -= 0.1

    def start_pan(self, event):
        self.spectrometr_canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.spectrometr_canvas.scan_dragto(event.x, event.y, gain=1)
        
    def cameraI(self,i):
        self.cameraIndex=i
        self.detector = MotionDetector(self.cameraIndex)
        
    def console_data(self,f):
        readable_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        self.console.insert(INSERT, f'{readable_time}: {f}\n')
        self.console.see("end")

    def create_task_list(self):
        for i, task in enumerate(self.tasks):
            frame = Frame(self.c2, bg=self.DGRAY)
            frame.grid(row=i, column=0, sticky="ew", pady=5)

            label = Label(frame, text=task["name"], bg=self.DGRAY, fg='lightgray')
            label.grid(row=0, column=0, padx=10)
            status_label = Label(frame, textvariable=task["status"], bg=self.DGRAY, fg='red')
            status_label.grid(row=0, column=1, padx=10)

            complete_button = CButton(frame, text="Complete", command=lambda t=task, sl=status_label: self.complete_task(t, sl))
            complete_button.grid(row=0, column=2, padx=10)

            options_button = CButton(frame, text="Options", command=lambda t=task: self.show_task_options(t))
            options_button.grid(row=0, column=3, padx=10)

    def complete_task(self, task,status_label):
        status_label.config(fg='green')
        task["status"].set("Completed")

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
            ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.EXPOSURE)
            assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
        elif task['name'] == "Auto white balance":
            ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.WHITE_SHADING)
            assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
        print(f"Options for {task['name']}")
    
    def draw_lines_on_image(self):
        if self.original_image is not None:
            image_array = np.array(self.original_image)
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_array
            gray_image = cv2.GaussianBlur(gray_image, (13, 13), 0)
            gray_image = (gray_image > 33).astype(np.uint8) * 255
            kernel = np.ones((20,20), np.uint8)
            dist = cv2.distanceTransform(gray_image, distanceType=cv2.DIST_L2, maskSize=5)

            # set up cross for tophat skeletonization
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            skeleton = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

            # threshold skeleton
            ret, skeleton = cv2.threshold(skeleton,0,255,0)
            #skeleton = cv2.GaussianBlur(skeleton, (11, 11), 0)
            #skeleton = (skeleton > 31).astype(np.uint8) * 255
            #skeleton = cv2.morphologyEx(cv2.GaussianBlur(skeleton, (11, 11), 0), cv2.MORPH_TOPHAT, kernel)
            #skeleton = cv2.threshold(skeleton,0,255,0)[1]
            self.image_tk = ImageTk.PhotoImage(Image.fromarray(skeleton))
            new_width = int(self.original_image.width * self.scale)
            new_height = int(self.original_image.height * self.scale)
            resized_image = self.original_image.resize((new_width, new_height))
            #self.image_tk = ImageTk.PhotoImage(resized_image)
            self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
        
    def update_sizes(self):
        root_width = self.window.winfo_width()
        root_height = self.window.winfo_height()

        frame_width = root_width // 2 -10
        frame_height = root_height // 4 -10

        self.console.config(y=self.winfo_y()+frame_height-10,width=frame_width,height=10)
        self.c1.config(width=frame_width, height=frame_height*2)
        self.c2.config(width=frame_width, height=frame_height*2)
        self.c3.config(width=frame_width, height=frame_height)
        self.c4.config(width=frame_width, height=frame_height)
        self.c5.config(width=frame_width, height=frame_height)
        
        self.c1.grid_propagate(False)
        self.c2.grid_propagate(False)
        self.c3.grid_propagate(False)
        self.c4.grid_propagate(False)
        self.c5.grid_propagate(False)
        
    def update_canvas(self):
        self.button_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def sequence(self,st):
        steps = st.split(',')
        for s in steps:
            s = s.split(':')
            self.move(s[0],s[1])
        
    def move(self,dir,step=100):
        self.measurements.append(len(self.measurements))
        self.draw_measurements()
        if self.connected:
            if dir == 'r':
                self.ports[0].write((f"M:1+P{step}\r\n").encode())
                time.sleep(1)
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'l':
                self.ports[0].write((f"M:1-P{-step}\r\n").encode())
                time.sleep(1)
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'u':
                self.ports[1].write((f"M:1+P{step}\r\n").encode())
                time.sleep(1)
                self.ports[1].write('G:\r\n'.encode())
            elif dir == 'd':
                self.ports[1].write((f"M:1-P{-step}\r\n").encode())
                time.sleep(1)
                self.ports[1].write('G:\r\n'.encode())
            elif dir == 'o':
                self.ports[0].write((f"H:1\r\n").encode())
                self.ports[1].write((f"H:1\r\n").encode())
            self.ports[0].write('Q:\r\n'.encode())
            self.ports[1].write('Q:\r\n'.encode())
            x = self.ports[0].readline().decode()
            y = self.ports[1].readline().decode()
            self.v.config(text=f'x:{x},y:{y}')

    def create_widgets(self):
        self.start_camera()
        self.start_spectrometr()
        self.spectrum()
        self.create_task_list()
        self.bind("<Configure>", lambda e: self.update_sizes())
        
        Label(self.c1, text="Detected Movement Direction:",background=self.DGRAY,fg='lightgray').grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction,background=self.DGRAY,fg='lightgray')
        self.direction_label.grid(row=0,column=1)

    def start_camera(self):
        if not self.calibrated:
            self.detector = MotionDetector(self.cameraIndex)
        else:
            self.detector = None
        self.update_video_feed()
    
    def analize(self):
        if self.spec:
            f=1
            W=1
            x=1
            F=1
            R=1
            r=1
            k=1
            theta=1
            I=1
            t=1
            lambda_ = np.pi/np.arcsin(np.sqrt((I*np.exp((-2*f**2*x**2)/(W**2*F**2))-(1-R*r)**2/(4*R*r))))/(2*t*np.cos(theta)-(2*t*np.sin(theta)*x)/F-(t*np.cos(theta)*x**2)/F**2)
            return lambda_
        
    def start_spectrometr(self):
        frame = np.zeros([1088,2048], dtype=np.uint8)
        ret = PxLApi.initialize(0)
        if not(PxLApi.apiSuccess(ret[0])):
            print("Error: Unable to initialize a camera! rc = %i" % ret[0])
            return 1

        self.hCamera = ret[1]
        ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)

        if PxLApi.apiSuccess(ret[0]):
            for i in range(1):            
                ret = self.get_next_frame(self.hCamera, frame, 5)
                self.spec = Image.fromarray(frame)
                image_tk = ImageTk.PhotoImage(image=self.spec)
                self.spectrometr_image = self.spectrometr_canvas.create_image(0, 0, anchor="nw", image=image_tk)

        PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
        assert PxLApi.apiSuccess(ret[0]), "setStreamState with StreamState.STOP failed"

        PxLApi.uninitialize(self.hCamera)
        assert PxLApi.apiSuccess(ret[0]), "uninitialize failed"
        return 0
    
    def get_next_frame(self,hCamera, frame, maxNumberOfTries):
        ret = (PxLApi.ReturnCode.ApiUnknownError,)
        for i in range(maxNumberOfTries):
            ret = PxLApi.getNextNumPyFrame(self.hCamera, frame)
            if PxLApi.apiSuccess(ret[0]):
                return ret
            else:
                if PxLApi.ReturnCode.ApiStreamStopped == ret[0] or \
                    PxLApi.ReturnCode.ApiNoCameraAvailableError == ret[0]:
                    return ret
                else:
                    print("    Hmmm... getNextFrame returned %i" % ret[0])
        return ret

    def update_video_feed(self):
        if self.detector:
            direction, frame = self.detector.detect_movement_direction()

            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.image = Image.fromarray(rgb_frame)
                # self.original_image = self.image
                new_width = int(self.original_image.width * self.scale)
                new_height = int(self.original_image.height * self.scale)
                resized_image = self.original_image.resize((new_width, new_height))
                self.image_tk = ImageTk.PhotoImage(resized_image)

                self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
                image_tk = ImageTk.PhotoImage(image=self.image)
                self.frame_label.configure(image=image_tk)
                self.frame_label.image = image_tk
                self.draw_lines_on_image()

            if direction:
                self.direction.set(f"Movement: {direction}")

        self.after(10, self.update_video_feed)
    def calibrate(self):
        pass
    def hotmap(self,i,n):
        HotmapWindow(self, i, n, self.image)
  
    def spectrum(self):
        fig, ax = plt.subplots(figsize=(5, 2),facecolor=self.DGRAY)
        ax.set_facecolor(self.DGRAY)
        x = np.linspace(-10, 10, 100)
        y = np.exp(-x**2)
        ax.plot(x,y,color='darkgreen')
        ax.grid()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        canvas = FigureCanvasTkAgg(fig, master=self.c4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=X, expand=True)
        plt.tight_layout()
        
    def draw_measurements(self):
        for i,n in enumerate(self.measurements):
            CButton(self.button_frame,text=f'{i}',command=lambda i=i, n=n: self.hotmap(i,n),width=2,height=1).grid(row=i // 40,column=i%40)
        self.update_canvas()
        
class HotmapWindow(CustomToplevel):
    def __init__(self, parent, i, n, image):
        CustomToplevel.__init__(self, parent)
        self.set_title(f'{i}')

        x = np.arange(0, image.width, 1)
        y = np.arange(0, image.height, 1)
        X, Y = np.meshgrid(x, y)
        Z1 = np.cos(X)
        Z2 = np.sin(Y)
        data = (Z1 - Z2) * 2
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=self.DGRAY)
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        cax = ax.imshow(data, cmap='hot', norm=norm)
        ax.set_xlabel('Oś X', color='white')
        ax.set_ylabel('Oś Y', color='white') 
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_ticks(cbar.get_ticks())
        cbar.ax.tick_params(labelcolor='white')
        img = image
        ax.imshow(img, alpha=0.5)
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.update_idletasks()
        plot_width = canvas.get_tk_widget().winfo_width()
        plot_height = canvas.get_tk_widget().winfo_height()
        self.geometry(f"{plot_width}x{plot_height + self.title_bar.winfo_height()}")

class Options(CustomToplevel):
    def __init__(self, parent):
        CustomToplevel.__init__(self, parent)
        self.geometry('500x400')
        self.set_title("Options")

        self.create_options()
        for w in self.winfo_children():
            w.config(bg=self.DGRAY,fg='lightgray',highlightbackground='white')
            for w in w.winfo_children():
                w.config(bg=self.DGRAY,fg='lightgray',highlightbackground='white')

    def create_options(self):
        Label(self.window, text=f"Options for {1}", bg=self.DGRAY, fg='lightgray').pack(pady=10)
        CButton(self.window, text="Import Settings", command=self.import_settings).pack(pady=10)

    def import_settings(self):
        global config
        file_path = filedialog.askopenfilename(title="Select Settings File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            config = json.load(open(file_path))
            print(f"Importing settings from {file_path}")
        self.focus()
        
if __name__ == "__main__":
    app = App()
    app.mainloop()