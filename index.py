from tkinter import *
from tkinter import ttk
from pixelinkWrapper import*
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from ctypes import*
import ctypes.wintypes
import threading
import win32api, win32con
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
import time
from libsonyapi.camera import Camera
from libsonyapi.actions import Actions


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

class App(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.state('zoomed')
        #self.geometry(f'{self.winfo_width}x{self.winfo_height}')
        self.configure(bg='#242424')
        
        self.c1 = LabelFrame(self,text='Camera',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c2 = LabelFrame(self,text='Controls',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c3 = LabelFrame(self,text='Spectrometr View',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c4 = LabelFrame(self,text='Specterum',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c5 = LabelFrame(self,text='Analyzed lasers',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.c2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.c3.grid(row=1, column=0, sticky="nsew", padx=5, pady=5, rowspan=2)
        self.c4.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.c5.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.update_sizes()
        self.bind("<Configure>", lambda event: self.update_sizes())
        
        self.canvas = Canvas(self.c5,bg='#242424')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar_x = ttk.Scrollbar(self.c5, orient="horizontal", command=self.canvas.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        scrollbar_y = ttk.Scrollbar(self.c5, orient="vertical", command=self.canvas.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        
        self.custom_scroll()

        self.canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)

        self.c5.grid_rowconfigure(0, weight=1)
        self.c5.grid_columnconfigure(0, weight=1)

        self.button_frame = Frame(self.canvas,bg='#242424')
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = range(100)
        self.data = range(100)
        self.speedX = 1
        self.speedY = 1
        
        self.draw_measurements()
        
        self.update_canvas()
        self.canvas.bind("<Configure>", lambda event: self.update_canvas())
        
        self.frame_label = Label(self.c1)
        self.frame_label.place(x=0,y=0)
        
        self.left = Button(self.c2,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.left.place(x=25,y=50)
        self.right = Button(self.c2,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.right.place(x=75,y=50)
        self.up = Button(self.c2,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.up.place(x=50,y=25)
        self.down = Button(self.c2,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.down.place(x=50,y=75)
        self.origin = Button(self.c2,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.origin.place(x=50,y=50)
        self.v = Label(self.c2)
        self.v.place(x=50,y=100)
        
        self.connected = False
        
        self.ports = []
        if len(list(serial.tools.list_ports.comports())) != 0:
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
        
        self.create_widgets()
        self.update_sizes()
        
    def update_sizes(self):
        self.state('zoomed')
        root_width = self.winfo_width()
        root_height = self.winfo_height()

        frame_width = root_width // 2 -10
        frame_height = root_height // 4 -10

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
        canvas_width = self.button_frame.winfo_width()
        canvas_height = self.button_frame.winfo_height()
        self.canvas.itemconfig(self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw"), width=canvas_width, height=canvas_height)
        
    def move(self,dir):
        if self.connected:
            if dir == 'r':
                self.ports[0].write((f"M:1+P{100}\r\n").encode())
                time.sleep(1)
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'l':
                self.ports[0].write((f"M:1-P{-100}\r\n").encode())
                time.sleep(1)
                self.ports[0].write('G:\r\n'.encode())
            elif dir == 'u':
                self.ports[1].write((f"M:1+P{100}\r\n").encode())
                time.sleep(1)
                self.ports[1].write('G:\r\n'.encode())
            elif dir == 'd':
                self.ports[1].write((f"M:1-P{-100}\r\n").encode())
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
        self.update_sizes()
        self.update()
        
        Label(self.c1, text="Detected Movement Direction:").grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction)
        self.direction_label.grid(row=0,column=1)
        
    def custom_scroll(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Horizontal.TScrollbar",
            gripcount=0,
            background='#3d3d3d',
            darkcolor='#242424',
            lightcolor='#242424',
            troughcolor='#242424',
            bordercolor='#242424',
            arrowcolor='#242424',
        )
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background='#3d3d3d',
            darkcolor='#242424',
            lightcolor='#242424',
            troughcolor='#242424',
            bordercolor='#242424',
            arrowcolor='#242424',
        )

    def start_camera(self):
        self.detector = MotionDetector()
        self.update_video_feed()
        
    def start_spectrometr(self):
        global hCamera
        global topHwnd
        ret = PxLApi.initialize(0)
        if PxLApi.apiSuccess(ret[0]):
            hCamera = ret[1]

            # Just use all of the camers's current settings.
            # Start the stream
            ret = PxLApi.setStreamState(hCamera, PxLApi.StreamState.START)
            if PxLApi.apiSuccess(ret[0]):

                # Step 3
                #      Start the preview / message pump, as well as the TkInter window resize handler
                topHwnd =  int(self.c3.frame(),0)

                self.start_preview()
                self.bind('<Configure>', self.winResizeHandler)
                
                # Step 4
                #      Call the start the UI -- it will only return on Window exit
                self.mainloop()

                # Step 5
                #      The user has quit the appliation, shut down the preview and stream
                previewState = PxLApi.PreviewState.STOP

                # Give preview a bit of time to stop
                time.sleep(0.05)
                
                PxLApi.setStreamState(hCamera, PxLApi.StreamState.STOP) 

            PxLApi.uninitialize(hCamera)
        else:
            Label(self.c3,text="No Camera Detected").grid(row=0,column=0,sticky='news')
    def winResizeHandler(self,event):
        global hCamera
        global topHwnd
        PxLApi.setPreviewSettings(hCamera, "", PxLApi.WindowsPreview.WS_VISIBLE | PxLApi.WindowsPreview.WS_CHILD , 0, 0, event.width, event.height, self.topHwnd)

    def start_preview(self):
        global previewState
        previewState = PxLApi.PreviewState.START
        previewThread = self.create_new_preview_thread()    
        previewThread.start()
        
    def create_new_preview_thread(self):
        return threading.Thread(target=self.control_preview_thread, args=(), daemon=True)
    
    def control_preview_thread(self):
        global hCamera
        global topHwnd
        user32 = windll.user32
        msg = ctypes.wintypes.MSG()
        pMsg = ctypes.byref(msg)
        
        # Create an arror cursor (see below)
        defaultCursor = win32api.LoadCursor(0,win32con.IDC_ARROW)
        
        # Get the current dimensions
        width = self.c3.winfo_width()
        height = self.c3.winfo_height()
        ret = PxLApi.setPreviewSettings(hCamera, "", PxLApi.WindowsPreview.WS_VISIBLE | PxLApi.WindowsPreview.WS_CHILD , 
                                        0, 0, width, height, topHwnd)

        # Start the preview (NOTE: camera must be streaming).  Keep looping until the previewState is STOPed
        ret = PxLApi.setPreviewState(hCamera, PxLApi.PreviewState.START)
        while (PxLApi.PreviewState.START == previewState and PxLApi.apiSuccess(ret[0])):
            if user32.PeekMessageW(pMsg, 0, 0, 0, 1) != 0:
                # All messages are simpy forwarded onto to other Win32 event handlers.  However, we do
                # set the cursor just to ensure that parent windows resize cursors do not persist within
                # the preview window
                win32api.SetCursor(defaultCursor)
                user32.TranslateMessage(pMsg)
                user32.DispatchMessageW(pMsg)
    
        # User has exited -- Stop the preview
        ret = PxLApi.setPreviewState(hCamera, PxLApi.PreviewState.STOP)
        assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
    def update_video_feed(self):
        if self.detector:
            direction, frame = self.detector.detect_movement_direction()

            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                image_tk = ImageTk.PhotoImage(image=image)

                self.frame_label.configure(image=image_tk)
                self.frame_label.image = image_tk

            if direction:
                self.direction.set(f"Movement: {direction}")

        self.after(10, self.update_video_feed)
    def calibrate(self):
        pass
    def hotmap(self,i,n):
        t = Toplevel()
        t.title(f'{i}')
        data = np.random.rand(100, 100)
        fig, ax = plt.subplots(figsize=(5, 5),facecolor='#242424')
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        cax = ax.imshow(data, cmap='hot', norm=norm)
        ax.set_xlabel('Oś X', color='white')
        ax.set_ylabel('Oś Y', color='white') 
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_ticks(cbar.get_ticks())
        cbar.ax.tick_params(labelcolor='white')
        img = Image.open('1.png')
        img = img.resize((data.shape[1], data.shape[0]))
        ax.imshow(img, alpha=0.5)
        canvas = FigureCanvasTkAgg(fig, master=t)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    def spectrum(self):
        fig, ax = plt.subplots(facecolor='#242424')
        ax.set_facecolor('#242424')
        x = np.linspace(-10, 10, 100)
        y = np.exp(-x**2)
        ax.plot(x,y,color='darkgreen')
        ax.grid()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        canvas = FigureCanvasTkAgg(fig, master=self.c4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=False)
    def draw_measurements(self):
        for i,n in enumerate(self.measurements):
            b = Button(self.button_frame,text=f'{i}',command=lambda: self.hotmap(i,n),width=2,height=1)
            b.grid(row=i // 10,column=i%40)

        
class Options(Toplevel):
    def __init__(self, parent, controller):
        Toplevel.__init__(self, parent)
        self.geometry('500x400')
        self.configure(bg='#242424')
        self.controller = controller
        x_frame = LabelFrame(self, text="X-Axis Controls",bg='#242424',fg='white')
        x_frame.place(x=10, y=10, width=220, height=120)
        
        Button(x_frame, text="Move Left", command=lambda: move_x_axis("left")).place(x=10, y=10, width=80, height=30)
        Button(x_frame, text="Move Right", command=lambda: move_x_axis("right")).place(x=100, y=10, width=80, height=30)

        Label(x_frame, text="Speed:").place(x=10, y=50)
        self.x_speed = Scale(x_frame, from_=0, to=100, orient="horizontal")
        self.x_speed.place(x=60, y=40, width=120, height=40)

        y_frame = LabelFrame(self, text="Y-Axis Controls",bg='#242424',fg='white')
        y_frame.place(x=250, y=10, width=220, height=120)

        Button(y_frame, text="Move Up", command=lambda: move_y_axis("up")).place(x=10, y=10, width=80, height=30)
        Button(y_frame, text="Move Down", command=lambda: move_y_axis("down")).place(x=100, y=10, width=80, height=30)

        Label(y_frame, text="Speed:").place(x=10, y=50)
        self.y_speed = Scale(y_frame, from_=0, to=100, orient="horizontal")
        self.y_speed.place(x=60, y=40, width=120, height=40)

        calib_frame = LabelFrame(self, text="Calibration",bg='#242424',fg='white')
        calib_frame.place(x=10, y=140, width=460, height=100)

        Button(calib_frame, text="Set Origin", command=set_origin).place(x=10, y=10, width=100, height=30)
        Button(calib_frame, text="Go to Origin", command=go_to_origin).place(x=120, y=10, width=100, height=30)

        Button(calib_frame, text="Jog X +1", command=lambda: jog("X", 1)).place(x=230, y=10, width=80, height=30)
        Button(calib_frame, text="Jog X -1", command=lambda: jog("X", -1)).place(x=320, y=10, width=80, height=30)

        Button(calib_frame, text="Jog Y +1", command=lambda: jog("Y", 1)).place(x=230, y=50, width=80, height=30)
        Button(calib_frame, text="Jog Y -1", command=lambda: jog("Y", -1)).place(x=320, y=50, width=80, height=30)

        pos_frame = LabelFrame(self, text="Position Control",bg='#242424',fg='white')
        pos_frame.place(x=10, y=250, width=460, height=80)

        Label(pos_frame, text="X:").place(x=10, y=10)
        self.x_entry = Entry(pos_frame, width=10)
        self.x_entry.place(x=40, y=10)

        Label(pos_frame, text="Y:").place(x=120, y=10)
        self.y_entry = Entry(pos_frame, width=10)
        self.y_entry.place(x=150, y=10)

        Button(pos_frame, text="Move to Position", command=lambda: move_to_position(self.x_entry.get(), self.y_entry.get())).place(x=240, y=10, width=150, height=30)

        Button(self, text="STOP", command=stop_motors).place(x=190, y=340, width=100, height=30)
        
def move_x_axis(direction):
    print(f"X-axis moving {direction}")

def move_y_axis(direction):
    print(f"Y-axis moving {direction}")

def stop_motors():
    print("Motors stopped")

def set_origin():
    print("Origin set to current position")

def go_to_origin():
    print("Moving to origin (0, 0)")

def jog(axis, step):
    print(f"Jogging {axis}-axis by {step} steps")

def move_to_position(x, y):
    print(f"Moving to position X: {x}, Y: {y}")
    

if __name__ == "__main__":
    global previewState
    global hCamera
    global topHwnd
    app = App()
    
    menu = Menu(app,bg='#242424')
    app.config(menu=menu)
    fileMenu = Menu(menu,tearoff=0)
    fileMenu.add_command(label="Options", command=lambda:Options(app.c1,app))
    menu.add_cascade(label="File", menu=fileMenu)
    app.mainloop()
