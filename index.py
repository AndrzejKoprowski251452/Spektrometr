from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np

class MotionDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
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
    def __init__(self):
        super().__init__()
        self.geometry("1200x800")
        self.state("zoomed")
        
        self.detector = None
        self.direction = StringVar(value="No movement detected")
        
        self.frame_label = Label(self)
        self.frame_label.grid(row=1,column=0)
        
        self.create_widgets()

    def create_widgets(self):
        self.start_button = Button(self, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0,column=0)
        
        Label(self, text="Detected Movement Direction:").grid(row=1,column=1)
        self.direction_label = Label(self, textvariable=self.direction)
        self.direction_label.grid(row=1,column=2)

    def start_camera(self):
        self.detector = MotionDetector(video_source=0)
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

if __name__ == "__main__":
    app = App()
    app.mainloop()
