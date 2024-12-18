from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
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
        self.geometry("1200x800")
        self.state("zoomed")
        
        self.container = Frame(self)
        self.container.grid()
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.frame_label = Label(self)
        self.frame_label.place(x=0,y=0)
        
        self.left = Button(self,text='←',width=2,height=1,command=lambda :self.move('l'))
        self.left.place(x=1000,y=100)
        self.right = Button(self,text='→',width=2,height=1,command=lambda :self.move('r'))
        self.right.place(x=1050,y=100)
        self.up = Button(self,text='↑',width=2,height=1,command=lambda :self.move('u'))
        self.up.place(x=1025,y=75)
        self.down = Button(self,text='↓',width=2,height=1,command=lambda :self.move('d'))
        self.down.place(x=1025,y=125)
        self.origin = Button(self,text='o',width=2,height=1,command=lambda:self.move('o'))
        self.origin.place(x=1025,y=100)
        self.v = Label(self)
        self.v.place(x=1025,y=150)
        
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
        else:
            self.connected = False
        
        self.create_widgets()
        
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
        
        Label(self, text="Detected Movement Direction:").grid(row=1,column=1)
        self.direction_label = Label(self, textvariable=self.direction)
        self.direction_label.grid(row=1,column=5)

    def start_camera(self):
        self.detector = MotionDetector()
        self.update_video_feed()

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
        
class Options(Toplevel):
    def __init__(self, parent, controller):
        Toplevel.__init__(self, parent)
        self.geometry('500x400')
        self.controller = controller
        x_frame = LabelFrame(self, text="X-Axis Controls")
        x_frame.place(x=10, y=10, width=220, height=120)
        
        Button(x_frame, text="Move Left", command=lambda: move_x_axis("left")).place(x=10, y=10, width=80, height=30)
        Button(x_frame, text="Move Right", command=lambda: move_x_axis("right")).place(x=100, y=10, width=80, height=30)

        Label(x_frame, text="Speed:").place(x=10, y=50)
        self.x_speed = Scale(x_frame, from_=0, to=100, orient="horizontal")
        self.x_speed.place(x=60, y=40, width=120, height=40)

        y_frame = LabelFrame(self, text="Y-Axis Controls")
        y_frame.place(x=250, y=10, width=220, height=120)

        Button(y_frame, text="Move Up", command=lambda: move_y_axis("up")).place(x=10, y=10, width=80, height=30)
        Button(y_frame, text="Move Down", command=lambda: move_y_axis("down")).place(x=100, y=10, width=80, height=30)

        Label(y_frame, text="Speed:").place(x=10, y=50)
        self.y_speed = Scale(y_frame, from_=0, to=100, orient="horizontal")
        self.y_speed.place(x=60, y=40, width=120, height=40)

        calib_frame = LabelFrame(self, text="Calibration")
        calib_frame.place(x=10, y=140, width=460, height=100)

        Button(calib_frame, text="Set Origin", command=set_origin).place(x=10, y=10, width=100, height=30)
        Button(calib_frame, text="Go to Origin", command=go_to_origin).place(x=120, y=10, width=100, height=30)

        Button(calib_frame, text="Jog X +1", command=lambda: jog("X", 1)).place(x=230, y=10, width=80, height=30)
        Button(calib_frame, text="Jog X -1", command=lambda: jog("X", -1)).place(x=320, y=10, width=80, height=30)

        Button(calib_frame, text="Jog Y +1", command=lambda: jog("Y", 1)).place(x=230, y=50, width=80, height=30)
        Button(calib_frame, text="Jog Y -1", command=lambda: jog("Y", -1)).place(x=320, y=50, width=80, height=30)

        pos_frame = LabelFrame(self, text="Position Control")
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
    
    menu = Menu(app,background='#F0F0F0')
    app.config(menu=menu)
    fileMenu = Menu(menu,tearoff=0)
    fileMenu.add_command(label="Options", command=lambda:Options(app.container,app))
    menu.add_cascade(label="File", menu=fileMenu)
    app.mainloop()
