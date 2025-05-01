from tkinter import *
from tkinter import ttk,filedialog
from pixelinkWrapper import*
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ctypes import*
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
import time
from addons import *
import threading

options = json.load(open('options.json'))

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
        self.image_tk = None
        self.scale = 1.0
        
        self.left = CButton(self.c1,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.right = CButton(self.c1,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.up = CButton(self.c1,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.down = CButton(self.c1,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.origin = CButton(self.c1,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.v = Label(self.c1,text='x:0,y:0',background=self.DGRAY,fg='lightgray',anchor='center')
        self.s = CButton(self.c1,text='s',width=2,height=1,command=self.sequence)
        
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
        self.s.place(x=100,y=50)
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
        
        self.tasks = [
            {"name": "Camera 1", "status": StringVar(value="Pending"),"event": None},
            {"name": "Camera 2", "status": StringVar(value="Pending"),"event": None},
            {"name": "Auto exposure", "status": StringVar(value="Not calibrated"),"event": None},
            {"name": "Auto white balance", "status": StringVar(value="Not calibrated"),"event": None},
        ]
        
        self.image = None  
        self.spec = None      
        self.connected = False
        self.calibrated = False
        
        self.ports = []
        self.connected = True
        if False and len(list(serial.tools.list_ports.comports())) != 0 and any(['COM' in p[0] for p in serial.tools.list_ports.comports()]):
            self.connected = True
            for i in serial.tools.list_ports.comports():
                s = serial.Serial(i[0])
                s.baudrate=9600
                s.BYTESIZES=serial.EIGHTBITS
                s.PARITIES=serial.PARITY_NONE
                s.STOPBITS=serial.STOPBITS_ONE
                s.timeout=1
                s.rtscts=True
                self.ports.append(s)
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
            frame = Frame(self.c2, bg=self.DGRAY)
            frame.grid(row=i, column=0, sticky="ew", pady=5)

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
            ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.EXPOSURE)
            assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
        elif task['name'] == "Auto white balance":
            ret = PxLApi.getFeature(self.hCamera, PxLApi.FeatureId.WHITE_SHADING)
            assert PxLApi.apiSuccess(ret[0]), "%i" % ret[0]
        print(f"Options for {task['name']}")
        
    def find_mods(self):
        self.original_image = Image.open("7.bmp")
        img = np.array(self.original_image.convert('L'))  # Convert to grayscale
        _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        points = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = 1/(cv2.mean(img, mask=mask)[0]/(w*h))

            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #cv2.line(img_color, (x + w // 2, 0), (x + w // 2, img_color.shape[0]), (0, 255, 0), 2)
            cv2.putText(img_color, f'{mean_val:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            points.append((x + w // 2, mean_val))

        self.original_image = Image.fromarray(img_color)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
        return points

    def draw_lines_on_image(self):
        if self.image_tk is None:
            image_array = np.array(self.original_image)
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_array
            gray_image = cv2.GaussianBlur(gray_image, (13, 13), 0)
            gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            grad = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=5)
            grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), grad)
            line_image = np.zeros_like(grad)
            contours, _ = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_positions = []
            for contour in contours:
                if len(contour) > 0:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    x1 = (box[0][0] + box[3][0]) // 2
                    y1 = (box[0][1] + box[3][1]) // 2
                    x2 = (box[1][0] + box[2][0]) // 2
                    y2 = (box[1][1] + box[2][1]) // 2
                    cv2.line(line_image, (x1, y1), (x2, y2), (255), 1)
                    line_positions.append((x1, y1, x2, y2))
            img = Image.fromarray(line_image)
            img = img.resize((int(self.original_image.width * self.scale), int(self.original_image.height * self.scale)))
            self.image_tk = ImageTk.PhotoImage(img)
            self.original_image = img
            self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
            return line_positions
        return []

    def map_lines_to_spectrum(self, points):
        fig, ax = plt.subplots(figsize=(5, 2), facecolor=self.DGRAY)
        ax.set_facecolor(self.DGRAY)
        x = np.linspace(0, 2048, 2048)
        y = np.zeros(2048)
        if points:
            for i in range(2048):
                if any(p[0] == i for p in points):
                    y[i] = next(p[1] for p in points if p[0] == i)
        ax.plot(x, y, color='darkgreen')
        ax.grid()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        canvas = FigureCanvasTkAgg(fig, master=self.c4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=X, expand=True)
        plt.tight_layout()

    def update_video_feed(self):
        if self.detector and not self.calibrated:
            direction, frame = self.detector.detect_movement_direction()
            if direction:
                self.direction.set(f"Movement: {direction}")
                if self.connected:
                    time.sleep(0.1)
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
            frame = self.detector.read()[1]
            frame = cv2.flip(frame, 1)
            if frame is not None:
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150,150), 1)  # Horizontal line
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150,150), 1)  # Vertical line

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                self.image = Image.fromarray(rgb_frame)
                resized_image = self.original_image.resize((int(self.original_image.width * self.scale), int(self.original_image.height * self.scale)))
                self.image_tk = ImageTk.PhotoImage(resized_image)

                self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
                image_tk = ImageTk.PhotoImage(image=self.image)
                self.frame_label.configure(image=image_tk)
                self.frame_label.image = image_tk

                if self.image_tk is None:
                    self.draw_lines_on_image()

        self.after(10, self.update_video_feed)
        
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
        
    def sequence(self):
        width = options['width']
        height = options['height']
        step_x = options['step_x']
        step_y = options['step_y']
        if self.connected:
            for i in range(0, width, step_x):
                self.move('r', i)
                for j in range(0, height, step_y):
                    self.move('d', j)
                    time.sleep(0.1)
        
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

    def create_widgets(self):
        self.start_camera()
        self.start_spectrometr()
        self.spectrum()
        self.create_task_list()
        self.bind("<Configure>", lambda e: self.update_sizes())
        self.find_mods()
        
        Label(self.c1, text="Detected Movement Direction:",background=self.DGRAY,fg='lightgray').grid(row=0,column=0)
        self.direction_label = Label(self.c1, textvariable=self.direction,background=self.DGRAY,fg='lightgray')
        self.direction_label.grid(row=0,column=1)

    def start_camera(self):
        if not self.calibrated:
            self.detector = MotionDetector(self.cameraIndex)
            #self.change_state(self.tasks[0], self.tasks[0]["status"])
        else:
            self.detector = None
        self.update_video_feed()
        
    def start_spectrometr(self):
        frame = np.zeros([1088,2048], dtype=np.uint8)
        ret = PxLApi.initialize(0)
        if not(PxLApi.apiSuccess(ret[0])):
            print("Error: Unable to initialize a camera! rc = %i" % ret[0])
            return 1
        else:
            pass

        self.hCamera = ret[1]
        ret = PxLApi.initialize(0, PxLApi.InitializeExFlags.ISSUE_STREAM_STOP)
        if PxLApi.apiSuccess(ret[0]):
            ret = PxLApi.loadSettings(self.hCamera, PxLApi.Settings.SETTINGS_FACTORY)
            if not PxLApi.apiSuccess(ret[0]):
                print("Could not load factory settings!")
            else:
                print("Factory default settings restored")

        ret = PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.START)

        if PxLApi.apiSuccess(ret[0]):
            for i in range(1):            
                ret = PxLApi.getNextNumPyFrame(self.hCamera, frame)
                self.spec = Image.fromarray(frame)
                image_tk = ImageTk.PhotoImage(image=self.spec)
                self.spectrometr_image = self.spectrometr_canvas.create_image(0, 0, anchor="nw", image=image_tk)

        PxLApi.setStreamState(self.hCamera, PxLApi.StreamState.STOP)
        assert PxLApi.apiSuccess(ret[0]), "setStreamState with StreamState.STOP failed"

        PxLApi.uninitialize(self.hCamera)
        assert PxLApi.apiSuccess(ret[0]), "uninitialize failed"
        return 0
    
    def spectrum(self):
        points = self.find_mods()
        self.map_lines_to_spectrum(points)
        
    def draw_measurements(self):
        for i,n in enumerate(self.measurements):
            CButton(self.button_frame,text=f'{i}',command=lambda i=i, n=n: HotmapWindow(self,i,n,self.image),width=2,height=1).grid(row=i // 40,column=i%40)
        self.update_canvas()

if __name__ == "__main__":
    app = App()
    app.mainloop()