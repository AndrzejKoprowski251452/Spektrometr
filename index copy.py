from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import csv
import glob


class ExampleApp(Tk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.geometry("1200x800")
        self.configure(bg="#222222")

        self.titlebar = Frame(self, bg="#111111", relief="raised", bd=0, height=32)
        self.titlebar.pack(fill=X, side=TOP)

        self.title_label = Label(self.titlebar, text="Przykładowa aplikacja z zakładkami", bg="#111111", fg="white", font=("Segoe UI", 12, "bold"))
        self.title_label.pack(side=LEFT, padx=10)

        self.close_btn = Button(self.titlebar, text="✕", bg="#111111", fg="white", bd=0, font=("Segoe UI", 12), command=self.destroy, activebackground="#c0392b", activeforeground="white")
        self.close_btn.pack(side=RIGHT, padx=5)
        self.min_btn = Button(
            self.titlebar, text="━", bg="#111111", fg="white", bd=0, font=("Segoe UI", 12),
            command=self.hide_window, activebackground="#444", activeforeground="white"
        )
        self.min_btn.pack(side=RIGHT)
        self.bind("<Control-Shift-M>", lambda e: self.deiconify())

        # Obsługa przesuwania okna
        self.titlebar.bind("<ButtonPress-1>", self.start_move)
        self.titlebar.bind("<B1-Motion>", self.do_move)
        self.title_label.bind("<ButtonPress-1>", self.start_move)
        self.title_label.bind("<B1-Motion>", self.do_move)

        # --- Stylizacja ttk ---
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook.Tab', background='#333333', foreground='white', font=('Segoe UI', 11))
        style.map('TNotebook.Tab', background=[('selected', '#444444')])
        style.layout("TNotebook.Tab",
    [('Notebook.tab', {'sticky': 'nswe', 'children':
        [('Notebook.padding', {'side': 'top', 'sticky': 'nswe', 'children':
            [('Notebook.label', {'side': 'top', 'sticky': ''})],
        })],
    })]
)

        # --- Zakładki ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=BOTH, expand=True)

        # Zakładki
        self.tab_camera = Frame(self.notebook, bg="#222222")
        self.tab_controls = Frame(self.notebook, bg="#222222")
        self.tab_spectro_view = Frame(self.notebook, bg="#222222")
        self.tab_spectrum = Frame(self.notebook, bg="#222222")
        self.tab_analyzed = Frame(self.notebook, bg="#222222")

        self.notebook.add(self.tab_camera, text="Camera")
        self.notebook.add(self.tab_controls, text="Controls")
        self.notebook.add(self.tab_spectro_view, text="Spectrometr View")
        self.notebook.add(self.tab_spectrum, text="Spectrum")
        self.notebook.add(self.tab_analyzed, text="Analyzed lasers")

        # --- Camera tab ---
        Label(self.tab_camera, text="Podgląd kamery", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        self.camera_label = Label(self.tab_camera, bg="#222222")
        self.camera_label.pack()
        # Przykładowy obrazek
        img = Image.new("RGB", (320, 240), color="gray")
        self.camera_img = ImageTk.PhotoImage(img)
        self.camera_label.config(image=self.camera_img)

        # --- Controls tab ---
        Label(self.tab_controls, text="Sterowanie", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        Button(self.tab_controls, text="Start", bg="#444", fg="white", font=('Segoe UI', 11), command=lambda: self.console_insert("Start!")).pack(pady=5)
        Button(self.tab_controls, text="Stop", bg="#444", fg="white", font=('Segoe UI', 11), command=lambda: self.console_insert("Stop!")).pack(pady=5)

        # Zakres X
        frame_x = Frame(self.tab_controls, bg="#222222")
        frame_x.pack(pady=10)
        Label(frame_x, text="X min:", bg="#222222", fg="lightgray", font=('Segoe UI', 10)).pack(side=LEFT)
        self.xmin_var = StringVar(value="0")
        Entry(frame_x, textvariable=self.xmin_var, width=8, bg="#333", fg="white", insertbackground="white").pack(side=LEFT, padx=5)
        Label(frame_x, text="X max:", bg="#222222", fg="lightgray", font=('Segoe UI', 10)).pack(side=LEFT)
        self.xmax_var = StringVar(value="2048")
        Entry(frame_x, textvariable=self.xmax_var, width=8, bg="#333", fg="white", insertbackground="white").pack(side=LEFT, padx=5)
        Button(frame_x, text="Ustaw zakres X", bg="#555", fg="white", font=('Segoe UI', 10), command=self.set_x_range).pack(side=LEFT, padx=10)

        # Konsola
        Label(self.tab_controls, text="Konsola:", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=(20,0))
        self.console = Text(self.tab_controls, bg="#111", fg="lime", height=8, font=('Consolas', 10))
        self.console.pack(fill=X, padx=10, pady=5)

        # --- Spectrometr View tab ---
        Label(self.tab_spectro_view, text="Widok spektrometru", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        self.spectro_canvas = Canvas(self.tab_spectro_view, bg="#333333", width=400, height=300, highlightthickness=0)
        self.spectro_canvas.pack(pady=10)
        # Przykładowy obraz
        self.spectro_img = ImageTk.PhotoImage(Image.new("L", (400, 300), color=128))
        self.spectro_canvas.create_image(0, 0, anchor="nw", image=self.spectro_img)

        # --- Spectrum tab ---
        Label(self.tab_spectrum, text="Wykres spektrum", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        self.fig, self.ax = plt.subplots(figsize=(5,2), facecolor="#222222")
        self.ax.set_facecolor("#222222")
        self.x = np.linspace(0, 2048, 2048)
        self.y = np.zeros(2048)
        (self.spectrum_line,) = self.ax.plot(self.x, self.y, color='lime')
        self.ax.set_xlabel("X (pixel)", color='white')
        self.ax.set_ylabel("Intensywność", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.set_title("Spektrum", color='white')
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig, master=self.tab_spectrum)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill=X, expand=True)

        # --- Analyzed lasers tab ---
        Label(self.tab_analyzed, text="Lista pomiarów (przyciski otwierają heatmapę)", bg="#222222", fg="white", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        self.measurements_frame = Frame(self.tab_analyzed, bg="#222222")
        self.measurements_frame.pack(fill=BOTH, expand=True)
        self.measurements = []
        self.load_measurements()
        self.draw_measurements()

    def start_move(self, event):
        self._x = event.x
        self._y = event.y

    def do_move(self, event):
        x = self.winfo_pointerx() - self._x
        y = self.winfo_pointery() - self._y
        self.geometry(f"+{x}+{y}")

    def console_insert(self, text):
        from time import strftime, localtime, time
        readable_time = strftime('%H:%M:%S', localtime(time()))
        self.console.insert(END, f'{readable_time}: {text}\n')
        self.console.see("end")

    def set_x_range(self):
        try:
            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            self.ax.set_xlim(xmin, xmax)
            self.spectrum_canvas.draw()
            self.console_insert(f"Zakres X ustawiony: {xmin} - {xmax}")
        except Exception as e:
            self.console_insert(f"Błąd ustawiania zakresu X: {e}")

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
                        self.console_insert(f"Pominięto wiersz z błędem: {row} ({e})")
            self.measurements.append(data)

    def draw_measurements(self):
        # Wyczyść stare przyciski
        for widget in self.measurements_frame.winfo_children():
            widget.destroy()
        for i, n in enumerate(self.measurements):
            Button(self.measurements_frame, text=f'Pomiar {i+1}', command=lambda i=i, n=n: self.show_heatmap(i, n)).pack(side=LEFT, padx=5, pady=5)

    def show_heatmap(self, idx, data):
        # Prosta heatmapa: suma wartości spektrum dla każdego punktu (x, y)
        import matplotlib.pyplot as plt
        import numpy as np
        xs = sorted(set([row[0] for row in data]))
        ys = sorted(set([row[1] for row in data]))
        nx, ny = len(xs), len(ys)
        heat = np.zeros((nx, ny))
        for row in data:
            ix = xs.index(row[0])
            iy = ys.index(row[1])
            heat[ix, iy] = sum(row[2])
        win = Toplevel(self)
        win.title(f"Heatmapa pomiaru {idx+1}")
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(heat, cmap="hot")
        fig.colorbar(im, ax=ax)
        ax.set_title("Heatmapa sumy spektrum")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def hide_window(self):
        self.withdraw()

if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()
