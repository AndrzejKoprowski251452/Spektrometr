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
from libsonyapi.camera import Camera
from libsonyapi.actions import Actions

class CustomWindow:
    def __init__(self, *args, **kwargs):
        self.tk_title = "Custom Window"
        self.LGRAY = '#424242'
        self.DGRAY = '#242424'
        self.RGRAY = '#444444'
        self.MGRAY = '#333333'

        self.title_bar = Frame(self, bg=self.RGRAY, relief='raised', bd=0, highlightthickness=1, highlightbackground=self.MGRAY, highlightcolor=self.MGRAY)
        
        self.close_button = Button(self.title_bar, text='  ×  ', command=self.destroy, bg=self.RGRAY, padx=2, pady=2, font=("calibri", 13), bd=0, fg='white', highlightthickness=0)
        self.minimize_button = Button(self.title_bar, text=' 🗕 ', command=self.minimize_me, bg=self.RGRAY, padx=2, pady=2, bd=0, fg='white', font=("calibri", 13), highlightthickness=0)
        self.title_bar_title = Label(self.title_bar, text=self.tk_title, bg=self.RGRAY, bd=0, fg='white', font=("helvetica", 10), highlightthickness=0)
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
            w.config(bg=self.DGRAY,fg='white',highlightbackground='white')
        
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
        )
        

class CustomTk(Tk, CustomWindow):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        CustomWindow.__init__(self, *args, **kwargs)
        self.tk_title = "Arczy Puszka"
        self.overrideredirect(True)
        self.config(bg=self.DGRAY, highlightthickness=0)
        self.state('zoomed')
        self.geometry(f'{self.winfo_width()}x{self.winfo_height()}')
        self.menu_button = Button(self.title_bar, text=' ≡ ', command=self.show_menu, bg=self.RGRAY, padx=2, pady=2, bd=0, fg='white', font=("calibri", 13), highlightthickness=0)
        self.menu_button.pack(side=LEFT,before=self.title_bar_title, ipadx=7, ipady=1)
        
        menu = Menu(self, tearoff=0)
        menu.config(bg=self.DGRAY,fg='white',relief='flat',bd=0,borderwidth=0,activebackground=self.RGRAY,activeforeground='white')
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
        
        sys.stdout = StreamToFunction(self.console_data)

        self.c1 = LabelFrame(self.window,text='Camera')
        self.c2 = LabelFrame(self.window,text='Controls')
        self.c3 = LabelFrame(self.window,text='Spectrometr View')
        self.c4 = LabelFrame(self.window,text='Specterum')
        self.c5 = LabelFrame(self.window,text='Analyzed lasers')
        
        self.console = Text(self.c2,background=self.DGRAY,fg='white',height=10,foreground='white')
        self.console.grid(row=10,column=0,sticky='ews',columnspan=10)
        
        self.canvas = Canvas(self.c5,bg=self.DGRAY)
        self.scrollbar_x = ttk.Scrollbar(self.c5, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y = ttk.Scrollbar(self.c5, orient="vertical", command=self.canvas.yview)
        self.button_frame = Frame(self.canvas,bg=self.DGRAY)
        
        self.frame_label = Label(self.c1)
        self.spectrometr_image = Label(self.c3,text='No camera detected')
        
        self.left = Button(self.c1,text='←',width=2,height=1,command=lambda :self.move('l'), bg=self.RGRAY, fg='white')
        self.right = Button(self.c1,text='→',width=2,height=1,command=lambda :self.move('r'), bg=self.RGRAY, fg='white')
        self.up = Button(self.c1,text='↑',width=2,height=1,command=lambda :self.move('u'), bg=self.RGRAY, fg='white')
        self.down = Button(self.c1,text='↓',width=2,height=1,command=lambda :self.move('d'), bg=self.RGRAY, fg='white')
        self.origin = Button(self.c1,text='o',width=2,height=1,command=lambda:self.move('o'), bg=self.RGRAY, fg='white')
        self.v = Label(self.c1,text='x:0,y:0',background=self.DGRAY,fg='white',anchor='center')
        
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
        self.spectrometr_image.place(x=0,y=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set,background=self.DGRAY)
        self.canvas.bind("<Configure>", lambda event: self.update_canvas())
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")
        self.c5.grid_rowconfigure(0, weight=1)
        self.c5.grid_columnconfigure(0, weight=1)
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = [1,2,3]
        self.data = range(100)
        self.speedX = 1
        self.speedY = 1
        
        self.tasks = [
            {"name": "Camera 1", "status": StringVar(value="Pending")},
            {"name": "Camera 2", "status": StringVar(value="Pending")},
            {"name": "Etalonu calibration", "status": StringVar(value="Pending")},
            {"name": "Grid calibration", "status": StringVar(value="Pending")},
            {"name": "Auto exposure", "status": StringVar(value="Pending")},
            {"name": "Auto white balance", "status": StringVar(value="Pending")}
        ]
        
        self.image = None        
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
        
    def console_data(self,f):
        self.console.insert(INSERT, f'{(time.time()):.2f}: {f}\n')
        self.console.see("end")

    def create_task_list(self):
        for i, task in enumerate(self.tasks):
            frame = Frame(self.c2, bg=self.DGRAY)
            frame.grid(row=i, column=0, sticky="ew", pady=5)

            label = Label(frame, text=task["name"], bg=self.DGRAY, fg='white')
            label.grid(row=0, column=0, padx=10)
            status_label = Label(frame, textvariable=task["status"], bg=self.DGRAY, fg='red')
            status_label.grid(row=0, column=1, padx=10)

            complete_button = Button(frame, text="Complete", command=lambda t=task, sl=status_label: self.complete_task(t, sl), bg=self.RGRAY, fg='white')
            complete_button.grid(row=0, column=2, padx=10)

            options_button = Button(frame, text="Options", command=lambda t=task: self.show_task_options(t), bg=self.RGRAY, fg='white')
            options_button.grid(row=0, column=3, padx=10)

    def complete_task(self, task,status_label):
        status_label.config(fg='green')
        task["status"].set("Completed")

    def show_task_options(self, task):
        print(f"Options for {task['name']}")
        
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
        
        Label(self.c1, text="Detected Movement Direction:").grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction)
        self.direction_label.grid(row=0,column=1)

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
            Button(self.button_frame,text=f'{i}',command=lambda i=i, n=n: self.hotmap(i,n),width=2,height=1, bg=self.RGRAY, fg='white').grid(row=i // 40,column=i%40)
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
            w.config(bg=self.DGRAY,fg='white',highlightbackground='white')
            for w in w.winfo_children():
                w.config(bg=self.DGRAY,fg='white',highlightbackground='white')

    def create_options(self):
        Label(self.window, text=f"Options for {1}", bg=self.DGRAY, fg='white').pack(pady=10)
        Button(self.window, text="Import Settings", command=self.import_settings, bg=self.RGRAY, fg='white').pack(pady=10)

    def import_settings(self):
        file_path = filedialog.askopenfilename(title="Select Settings File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            print(f"Importing settings from {file_path}")
        self.focus()
        
if __name__ == "__main__":
    app = App()
    app.mainloop()