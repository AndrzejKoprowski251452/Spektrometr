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
import asyncio
from async_tkinter_loop import async_mainloop
import os
import csv
import glob

options = json.load(open('options.json'))

class App(CustomTk):
    def __init__(self, *args, **kwargs):
        CustomTk.__init__(self,*args, **kwargs)
        self.state('zoomed')
        self.set_title("Arczy Puszka")
        self.iconbitmap("icon.ico")
        
        sys.stdout = StreamToFunction(self.console_data)
        
        self.options_window = None

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
        
        self.original_image = Image.open("3.bmp")
        self.image_tk = None 
        self.spectrometr_image = self.spectrometr_canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
        self.scale = 1.0
        
        self.left = CButton(self.c1,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.right = CButton(self.c1,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.up = CButton(self.c1,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.down = CButton(self.c1,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.origin = CButton(self.c1,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.v = Label(self.c1,text='x:0,y:0',background=self.DGRAY,fg='lightgray',anchor='center')
        self.s = CButton(self.c1,text='s',width=2,height=1,command=self.start_sequence)
        
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
        
        self.cameraIndex=0
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.measurements = []
        self.bin = []
        
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
            
        self.update_colors()
        self.draw_measurements()
        
        self.bind("<Configure>",lambda e: self.update_sizes())
        self.update_sizes()
        
        self.spectrometr_canvas.bind("<MouseWheel>", self.zoom)
        self.spectrometr_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.spectrometr_canvas.bind("<B1-Motion>", self.pan)
        
        self.xmin_var = StringVar(value=options['xmin'])
        self.xmax_var = StringVar(value=options['xmax'])
        
        Label(self.c2, text="X min:", bg=self.DGRAY, fg='lightgray').grid(row=2, column=0, sticky="w", padx=5)
        self.xmin_entry = Entry(self.c2, textvariable=self.xmin_var, width=8)
        self.xmin_entry.grid(row=2, column=1, sticky="w", padx=5)

        Label(self.c2, text="X max:", bg=self.DGRAY, fg='lightgray').grid(row=3, column=0, sticky="w", padx=5)
        self.xmax_entry = Entry(self.c2, textvariable=self.xmax_var, width=8)
        self.xmax_entry.grid(row=3, column=1, sticky="w", padx=5)

        CButton(self.c2, text="Ustaw zakres X", command=self.set_x_range).grid(row=4, column=0, columnspan=2, pady=5)
        
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

            if i == 0:
                Label(frame, text="X min:", bg=self.DGRAY, fg='lightgray').grid(row=0, column=9, padx=(30,5))
                self.xmin_entry = Entry(frame, textvariable=self.xmin_var, width=8)
                self.xmin_entry.grid(row=0, column=10, padx=5)

                Label(frame, text="X max:", bg=self.DGRAY, fg='lightgray').grid(row=0, column=11, padx=5)
                self.xmax_entry = Entry(frame, textvariable=self.xmax_var, width=8)
                self.xmax_entry.grid(row=0, column=12, padx=5)

                CButton(frame, text="Ustaw zakres X", command=self.set_x_range).grid(row=0, column=13, padx=10)

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
    #     self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)

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
    #         self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
    #         return line_positions
    #     return []

    def create_spectrum_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 2), facecolor=self.DGRAY)
        self.ax.set_facecolor(self.DGRAY)
        self.x = np.linspace(0, 2048, 2048)
        self.y = np.zeros(2048)
        (self.spectrum_line,) = self.ax.plot(self.x, self.y, color='darkgreen')
        self.ax.grid()
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig, master=self.c4)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill=X, expand=True)
        plt.tight_layout()

    def update_spectrum_plot(self):
        img = np.array(self.original_image.convert('L'))
        self.y = img.sum(axis=0)/img.shape[0]
        self.spectrum_line.set_ydata(self.y)
        y_max = max(self.y)
        self.ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1)
        self.spectrum_canvas.draw()
        
    async def update_video_feed(self):
        while True:
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
                frame = self.detector.read()[1]
                frame = cv2.flip(frame, 1)
                if frame is not None:
                    height, width, _ = frame.shape
                    center_x, center_y = width // 2, height // 2
                    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150,150), 1)  # Horizontal line
                    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150,150), 1)  # Vertical line

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.image = Image.fromarray(rgb_frame)
            image_tk = ImageTk.PhotoImage(image=self.image)
            self.frame_label.configure(image=image_tk)
            self.frame_label.image = image_tk
            frame = np.zeros([1088,2048], dtype=np.uint8)
            if self.hCamera is not None:
                ret = PxLApi.getNextNumPyFrame(self.hCamera, frame)
                self.spec = Image.fromarray(frame)
                self.original_image = self.spec
                resized_image = self.original_image.resize((int(self.original_image.width * self.scale), int(self.original_image.height * self.scale)))
                self.image_tk = ImageTk.PhotoImage(resized_image)
                self.spectrometr_canvas.itemconfig(self.spectrometr_image, image=self.image_tk)
                self.update_spectrum_plot()
            await asyncio.sleep(0.01)
                  
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
            self.draw_measurements()
            spectra.clear()
            self.move('o')

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
            CButton(self.button_frame,text=f'{i}',command=lambda i=i, n=n: HeatMapWindow(self,i,n,self.image),width=2,height=1).grid(row=i // 40,column=i%40)
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
                     
if __name__ == "__main__":
    app = App()
    app.after_idle(app.async_loop)
    app.after_idle(app.update_sizes)
    async_mainloop(app)
    if app.hCamera is not None:
        if PxLApi.StreamState == PxLApi.StreamState.START:
            PxLApi.setStreamState(app.hCamera, PxLApi.StreamState.STOP)
            PxLApi.uninitialize(app.hCamera)
