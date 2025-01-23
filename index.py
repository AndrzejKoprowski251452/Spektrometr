from tkinter import *
from tkinter import ttk
from pixelinkWrapper import*
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from ctypes import*
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
        self.tk_title = "Arczy Puszka"
        
        self.LGRAY = '#424242'
        self.DGRAY = '#242424'
        self.RGRAY = '#444444'

        self.title(self.tk_title) 
        self.overrideredirect(True)
        self.config(bg=self.DGRAY,highlightthickness=0)
        #self.iconbitmap("your_icon.ico") # to show your own icon 
        
        self.title_bar = Frame(self, bg=self.RGRAY, relief='raised', bd=0,highlightthickness=0)
        
        self.close_button = Button(self.title_bar, text='  ×  ', command=self.destroy,bg=self.RGRAY,padx=2,pady=2,font=("calibri", 13),bd=0,fg='white',highlightthickness=0)
        self.minimize_button = Button(self.title_bar, text=' 🗕 ',command=self.minimize_me,bg=self.RGRAY,padx=2,pady=2,bd=0,fg='white',font=("calibri", 13),highlightthickness=0)
        self.title_bar_title = Label(self.title_bar, text=self.tk_title, bg=self.RGRAY,bd=0,fg='white',font=("helvetica", 10),highlightthickness=0)
        
        self.window = Frame(self, bg=self.DGRAY,highlightthickness=0)
        
        self.title_bar.pack(fill=X)
        self.close_button.pack(side=RIGHT,ipadx=7,ipady=1)
        self.minimize_button.pack(side=RIGHT,ipadx=7,ipady=1)
        self.title_bar_title.pack(side=LEFT, padx=10)
        self.window.pack(expand=1, fill=BOTH)

        self.close_button.bind('<Enter>',lambda e:self.changex_on_hovering())
        self.close_button.bind('<Leave>',lambda e:self.returnx_to_normalstate())

        self.bind("<Expose>",lambda e:self.deminimize())
        self.after(10, lambda: self.set_appwindow())
        
        self.c1 = LabelFrame(self.window,text='Camera',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c2 = LabelFrame(self.window,text='Controls',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c3 = LabelFrame(self.window,text='Spectrometr View',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c4 = LabelFrame(self.window,text='Specterum',bg='#242424',highlightbackground='#ffffff',fg='white')
        self.c5 = LabelFrame(self.window,text='Analyzed lasers',bg='#242424',highlightbackground='#ffffff',fg='white')
        
        self.canvas = Canvas(self.c5,bg='#242424')
        self.scrollbar_x = ttk.Scrollbar(self.c5, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y = ttk.Scrollbar(self.c5, orient="vertical", command=self.canvas.yview)
        self.button_frame = Frame(self.canvas,bg='#242424')
        
        self.frame_label = Label(self.c1)
        self.spectrometr_image = Label(self.c3,text='No camera detected')
        
        self.left = Button(self.c2,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.right = Button(self.c2,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.up = Button(self.c2,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.down = Button(self.c2,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.origin = Button(self.c2,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.v = Label(self.c2)
        
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
        self.v.place(x=50,y=100)
        self.frame_label.place(x=0,y=0)
        self.spectrometr_image.place(x=0,y=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")
        self.canvas.bind("<Configure>", lambda event: self.update_canvas())
        self.c5.grid_rowconfigure(0, weight=1)
        self.c5.grid_columnconfigure(0, weight=1)
        
        self.update_sizes()
        self.bind("<Configure>", lambda e: self.update_sizes())
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = [1,2,3]
        self.data = range(100)
        self.speedX = 1
        self.speedY = 1
        
        self.image = None        
        self.connected = False
        self.calibrated = False
        
        self.ports = []
        if len(list(serial.tools.list_ports.comports())) != 0 and False:
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
        
        self.draw_measurements()
        self.custom_scroll()
        self.update_canvas()
        self.create_widgets()
        self.update_sizes()
        
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
        self.state('zoomed')
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
        
        
    def update_sizes(self):
        root_width = self.window.winfo_width()
        root_height = self.window.winfo_height()

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
        
    def sequence(self,st):
        steps = st.split(',')
        for s in steps:
            s = s.split(':')
            self.move(s[0],s[1])
        
    def move(self,dir,step=100):
        self.measurements.append(len(self.measurements))
        self.draw_measurements()
        self.update_canvas()
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
        
        Label(self.c1, text="Detected Movement Direction:").grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction)
        self.direction_label.grid(row=0,column=1)
        self.state('zoomed')
        self.update()
        self.geometry(f'{self.winfo_width()}x{self.winfo_height()}')
        
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
        if not self.calibrated:
            self.detector = MotionDetector()
        else:
            self.detector = None
        self.update_video_feed()
        
    def start_spectrometr(self):
        frame = np.zeros([1088,2048], dtype=np.uint8)
        ret = PxLApi.initialize(0)
        if not(PxLApi.apiSuccess(ret[0])):
            print("Error: Unable to initialize a camera! rc = %i" % ret[0])
            return 1

        hCamera = ret[1]
        ret = PxLApi.setStreamState(hCamera, PxLApi.StreamState.START)

        if PxLApi.apiSuccess(ret[0]):
            for i in range(1):            
                ret = self.get_next_frame(hCamera, frame, 5)
                print(frame.size)
                image = Image.fromarray(frame)
                image_tk = ImageTk.PhotoImage(image=image)
                self.spectrometr_image.configure(image=image_tk)
                self.spectrometr_image.image = image_tk

        PxLApi.setStreamState(hCamera, PxLApi.StreamState.STOP)
        assert PxLApi.apiSuccess(ret[0]), "setStreamState with StreamState.STOP failed"

        PxLApi.uninitialize(hCamera)
        assert PxLApi.apiSuccess(ret[0]), "uninitialize failed"
        return 0
    
    def get_next_frame(self,hCamera, frame, maxNumberOfTries):
        ret = (PxLApi.ReturnCode.ApiUnknownError,)
        for i in range(maxNumberOfTries):
            ret = PxLApi.getNextNumPyFrame(hCamera, frame)
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
                image_tk = ImageTk.PhotoImage(image=self.image)

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
        x = np.arange(0,self.image.width, 1)
        y = np.arange(0,self.image.height, 1)
        X, Y = np.meshgrid(x, y)
        Z1 = np.cos(X)
        Z2 = np.sin(Y)
        data = (Z1 - Z2) * 2
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
        img = self.image
        #img = img.resize((data.shape[1], data.shape[0]))
        ax.imshow(img, alpha=0.5)
        canvas = FigureCanvasTkAgg(fig, master=t)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    def spectrum(self):
        fig, ax = plt.subplots(figsize=(5, 5),facecolor='#242424')
        ax.set_facecolor('#242424')
        x = np.linspace(-10, 10, 100)
        y = np.exp(-x**2)
        ax.plot(x,y,color='darkgreen')
        ax.grid()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        canvas = FigureCanvasTkAgg(fig, master=self.c4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.tight_layout()
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
    app = App()
    
    #menu = Menu(app.window, background='#242424', fg='#242424')
    #fileMenu = Menu(menu,tearoff=0, background='#424242', foreground='black')
    #fileMenu.add_command(label="Options", command=lambda:Options(app,app))
    #app.config(menu=menu)
    #menu.add_cascade(label="File", menu=fileMenu)
    app.mainloop()
