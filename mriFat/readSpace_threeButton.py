import os
import threading
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, font, ttk
import nibabel as nib
from PIL import Image, ImageTk
from scipy import ndimage as ndi

import t1_seg
import dixon_local_imat   # default IMAT contrast for Dixon
import parallel


from abd_seg import segment_abdomen_stack, relabel_vat
# DICOM loading lives in dicom_io; re-exported here for scripts that import it from this file
from dicom_io import (readDicomFile, parseDicomFolder, MultipleSeriesError, SeriesChoiceCancelled,
                      dicom_affine, slice_spacing_mm)
import pipeline

# Thigh SAT Threshold slider, per sequence: Dixon fat (_F) raw signal (seg.segment_stack sat_thr),
# T1 TSE fat fraction of the local fat signal (t1_seg.SAT_FRACTION).
THIGH_SAT = {'DIXON_F': dict(default=100, from_=50, to=200, resolution=1, label="SAT Threshold (signal):"),
             'T1_TSE': dict(default=t1_seg.SAT_FRACTION, from_=0.30, to=0.60, resolution=0.01,
                            label="SAT Threshold (fat fraction):")}

# The GUI icons (sat.png, thigh.png, polygon.png, brush.png) sit in the project folder,
# one level above this file, wherever the app is started from
ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


def load_icon(name, size):
    return ImageTk.PhotoImage(Image.open(os.path.join(ICON_DIR, name)).resize((size, size), Image.Resampling.LANCZOS))

class ThreeButtonSlider(tk.Frame):
    def __init__(self, master, min_val=0, max_val=100, initial_min=20, initial_current=50, initial_max=80, width=300, height=50, parent=None):
        super().__init__(master)
        self.parent = parent if parent is not None else master
        self.min_val = min_val
        self.max_val = max_val
        self.current_min = initial_min
        self.current_value = initial_current
        self.current_max = initial_max
        self.width = width
        self.height = height
        
        # Create main frame
        self.frame = tk.Frame(master)
        
        # Create canvas for the slider
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg='white', highlightthickness=1)
        self.canvas.pack()
        self.canvas.focus_set()
        
        # Slider properties
        self.slider_height = 10
        self.button_radius = 8
        self.margin = 20
        self.slider_width = width - 2 * self.margin
        
        # Track which button is being dragged
        self.dragging = None
        self.drag_offset = 0
        
        # Create labels
        self.label_frame = tk.Frame(self.frame)
        self.label_frame.pack(fill='x', pady=5)
        
        self.min_label = tk.Label(self.label_frame, text=f"Min: {int(self.current_min)}")
        self.min_label.pack(side='left', padx=5)
        
        self.current_label = tk.Label(self.label_frame, text=f"Current: {int(self.current_value)}")
        self.current_label.pack(side='left', padx=5)
        
        self.max_label = tk.Label(self.label_frame, text=f"Max: {int(self.current_max)}")
        self.max_label.pack(side='right', padx=5)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
        # Initial draw
        self.draw_slider()
    
    def value_to_x(self, value):
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.margin + ratio * self.slider_width
    
    def x_to_value(self, x):
        ratio = (x - self.margin) / self.slider_width
        value = self.min_val + ratio * (self.max_val - self.min_val)
        value = round(value)
        return max(self.min_val, min(self.max_val, value))
    
    def draw_slider(self):
        self.canvas.delete('all')
        track_y = self.height // 2
        self.canvas.create_rectangle(
            self.margin, track_y - self.slider_height // 2,
            self.margin + self.slider_width, track_y + self.slider_height // 2,
            fill='lightgray', outline='gray'
        )
        min_x = self.value_to_x(self.current_min)
        max_x = self.value_to_x(self.current_max)
        self.canvas.create_rectangle(
            min_x, track_y - self.slider_height // 2,
            max_x, track_y + self.slider_height // 2,
            fill='lightblue', outline='blue'
        )
        self.canvas.create_oval(
            min_x - self.button_radius, track_y - self.button_radius,
            min_x + self.button_radius, track_y + self.button_radius,
            fill='blue', outline='darkblue', width=2, tags='min_button'
        )
        current_x = self.value_to_x(self.current_value)
        self.canvas.create_oval(
            current_x - self.button_radius, track_y - self.button_radius,
            current_x + self.button_radius, track_y + self.button_radius,
            fill='green', outline='darkgreen', width=2, tags='current_button'
        )
        self.canvas.create_oval(
            max_x - self.button_radius, track_y - self.button_radius,
            max_x + self.button_radius, track_y + self.button_radius,
            fill='red', outline='darkred', width=2, tags='max_button'
        )
    
    def on_click(self, event):
        min_x = self.value_to_x(self.current_min)
        current_x = self.value_to_x(self.current_value)
        max_x = self.value_to_x(self.current_max)
        # print(f"Click at x={event.x}, min_x={min_x}, current_x={current_x}, max_x={max_x}")
        if abs(event.x - current_x) <= self.button_radius:
            self.dragging = 'current'
            self.drag_offset = event.x - current_x
            # print("Dragging set to 'current'")
        elif abs(event.x - min_x) <= self.button_radius:
            self.dragging = 'min'
            self.drag_offset = event.x - min_x
            # print("Dragging set to 'min'")
        elif abs(event.x - max_x) <= self.button_radius:
            self.dragging = 'max'
            self.drag_offset = event.x - max_x
            # print("Dragging set to 'max'")
        else:
            print("No button clicked")
    
    def on_drag(self, event):
        if self.dragging:
            new_x = event.x - self.drag_offset
            new_value = self.x_to_value(new_x)
            if self.dragging == 'min':
                self.current_min = min(new_value, self.current_value)
            elif self.dragging == 'current':
                self.current_value = max(self.current_min, min(new_value, self.current_max))
            elif self.dragging == 'max':
                self.current_max = max(new_value, self.current_value)
            self.update_labels()
            self.draw_slider()
    
    def on_release(self, event):
        if self.dragging == 'current':
            try:
                if hasattr(self.parent, 'update_image_slice'):
                    self.parent.update_image_slice()
                else:
                    print("Parent does not have update_image_slice")
            except Exception as e:
                print(f"Error calling update_image_slice: {str(e)}")
        self.dragging = None
        self.drag_offset = 0
    
    def update_labels(self):
        self.min_label.config(text=f"Min: {int(self.current_min)}")
        self.current_label.config(text=f"Current: {int(self.current_value)}")
        self.max_label.config(text=f"Max: {int(self.current_max)}")
    
    def get_values(self):
        return int(self.current_min), int(self.current_value), int(self.current_max)
    
    def set_values(self, min_val, current_val, max_val):
        self.current_min = min_val
        self.current_max = max_val
        self.current_value = current_val
        if self.current_min > self.current_max:
            self.current_min, self.current_max = self.current_max, self.current_min
        self.update_labels()
        self.draw_slider()

    def set_values2(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.draw_slider()
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        self.frame.grid(**kwargs)
    
    def place(self, **kwargs):
        self.frame.place(**kwargs)

def line_plane_intersection(plane_xdir, plane_ydir, plane_point, ray_origin, ray_end, epsilon=1e-6):
    plane_xdir = np.array(plane_xdir)
    plane_ydir = np.array(plane_ydir)
    plane_point = np.array(plane_point)
    ray_origin = np.array(ray_origin)
    ray_end = np.array(ray_end)
    plane_normal = np.cross(plane_xdir, plane_ydir)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    ray = ray_end - ray_origin
    ray_direction = ray / np.linalg.norm(ray)
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        return 0
    w = np.dot(ray_end - ray_origin, plane_normal)
    r = np.dot(plane_point - ray_origin, plane_normal) / w
    p = ray_origin + r * (ray_end - ray_origin)
    y = np.dot(p - ray_origin, ray_direction)
    return y 

def normal_to_view_angles(normal):
    normal = normal / np.linalg.norm(normal)
    n_x, n_y, n_z = normal
    elev = np.degrees(np.arctan2(n_z, np.sqrt(n_x**2 + n_y**2)))
    azim = np.degrees(np.arctan2(n_y, n_x))
    return elev, azim

def normalize(slice_data):
    return (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))

class DicomViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dicom Viewer GUI")
        self.root.configure(background='black')
        self.click_state = 1  # Start with SAT mode (state 1)
        self.threshold = 0
        self.fat_fraction = 0.4   # abdomen: VAT fat threshold as a fraction of the measured fat signal
        self.sat_fraction = 0.4   # abdomen: SAT fat threshold, same units (the SAT/VAT boundary is fixed)
        self.ai_seg_active = False  # segmentation comes from AI Seg (or Combined on top of it)
        # Segmentation array layout: False (H, W, slices) like the images (Thigh Seg, SAT/AVT Seg),
        # True (W, H, slices) (AI Seg, Combined), None unknown (loaded file: guessed from the shape)
        self.seg_transposed = None
        self.ai_model = None       # thigh U-Net, loaded on the first AI Seg click
        self.ai_model_path = None
        self.cache = None
        self.dicom_long_info = None
        self.dicom_long_pixels = None
        self.dicom_short_info = None
        self.dicom_short_pixels = None
        self.short_sequence = None  # 'DIXON_F' or 'T1_TSE', set when short axis slices are loaded
        self.abd_state = None       # abdomen: fat signal + SAT/VAT boundary of the last SAT/AVT Seg (for VAT relabelling)
        self.abd_labels = None      # abdomen: the labels that SAT/AVT Seg (or the last relabelling) produced
        self.imat_contrast = dixon_local_imat.CONTRAST   # thigh: IMAT contrast setting (default per sequence)
        self.thigh_state = None     # thigh: first-pass labels + contrast maps of the last Thigh Seg (for IMAT relabelling)
        self.thigh_sat = THIGH_SAT['DIXON_F']['default']   # thigh: SAT threshold (units per sequence, see THIGH_SAT)
        self.thigh_radius = int(t1_seg.LOCAL_RADIUS_MM)    # thigh, T1: IMAT check neighbourhood (mm)
        self.thigh_sat_radius = int(t1_seg.SAT_RADIUS_MM)  # thigh, T1: SAT check neighbourhood (mm), 0 = off
        self.thigh_labels = None    # thigh: the labels that Thigh Seg (or the last relabelling) produced
        self.after_id = None  # To store after callback ID
        self.flip_axes = False  # Track whether axes are flipped
        
        # Main frame setup
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Left frame
        left_frame = tk.Frame(main_frame, width=200, bg="lightgray")
        left_frame.grid(row=0, column=0, sticky="ns")
        main_frame.grid_columnconfigure(0, weight=0)
        
        # Load images
        try:
            self.sat_img = load_icon("sat.png", 180)
            self.thigh_img = load_icon("thigh.png", 180)
        except Exception as e:
            # print(f"Error loading images: {str(e)}")
            self.sat_img = self.thigh_img = None
        
        self.switch_button = tk.Button(left_frame, image=self.sat_img, text="click to choose", command=self.update_button_image)
        self.switch_button.pack(pady=5)

        # Running-job status: a popup with a timer and progress bar while a segmentation runs
        self.busy = False
        self.busy_after_id = None
        
        # Sub-frames
        self.left_top_frame = tk.Frame(left_frame, bg="#FFF0F0", width=180, height=160)
        self.left_top_frame.pack(padx=5, pady=5)
        self.left_top_frame.pack_propagate(False)
        
        self.left_mid_frame = tk.Frame(left_frame, bg="#F0F0FF", width=180, height=200)
        self.left_mid_frame.pack(padx=5, pady=5)
        self.left_mid_frame.pack_propagate(False)
        
        self.left_bottom_frame = tk.Frame(left_frame, bg="#F0FFF0", width=180, height=80)
        self.left_bottom_frame.pack(padx=5, pady=5)
        self.left_bottom_frame.pack_propagate(False)

        # Threshold input frame
        self.threshold_frame = tk.Frame(left_frame, bg="#FFF8DC", width=180, height=150)
        self.threshold_frame.pack(padx=5, pady=5)
        self.threshold_frame.pack_propagate(False)

        # Buttons
        self.load_short_button = tk.Button(self.left_top_frame, text="Load short axis slices", command=self.load_short_files, width=16)
        self.load_long_button = tk.Button(self.left_top_frame, text="Load long axis slices", command=self.load_long_files, width=16)
        self.load_seg_button = tk.Button(self.left_top_frame, text="Load seg file (nii.gz)", command=self.load_segmentation_files, width=16)
        self.flip_button = tk.Button(self.left_top_frame, text="Flip Seg X/Y", command=self.flip_xy_axes, bg="#FF8C00", fg="white", width=16)
        self.auto_seg_button = tk.Button(self.left_mid_frame, text="SAT/AVT Seg", command=self.auto_segment, bg="#A20541", fg="white", width=16)
        self.show_image_button = tk.Button(self.left_bottom_frame, text="Show Image", command=self.show_image_popup, width=16)
        self.save_image_button = tk.Button(self.left_bottom_frame, text="Save Segmentation", command=self.save_segmentation, width=16)
        self.draw_line_button = tk.Button(self.left_mid_frame, text="Draw Line", command=self.activate_line_drawing, bg="#006503", fg="white", bd=2, relief="raised", width=16)
        self.thigh_button = tk.Button(self.left_mid_frame, text="Thigh Seg", command=self.thigh_mode, bg="#006503", fg="white", width=16)
        self.ai_seg_button = tk.Button(self.left_mid_frame, text="AI Seg", command=self.ai_seg_mode, bg="#4A0082", fg="white", width=16)
        self.swap_xy_button = tk.Button(self.left_mid_frame, text="Swap XY", command=self.swap_xy_segmentation, bg="#555555", fg="white", width=16)
        self.combined_seg_button = tk.Button(self.left_mid_frame, text="Combined", command=self.combined_seg_mode, bg="#8B4513", fg="white", width=16)

        # VAT fat fraction slider (abdomen): a pixel inside the SAT boundary is VAT if its signal is at
        # least this share of the fat signal.
        threshold_label = tk.Label(self.threshold_frame, text="VAT Fat Fraction:", bg="#FFF8DC", font=("Arial", 9))
        threshold_label.pack(pady=(3, 0))

        self.threshold_var = tk.DoubleVar(value=self.fat_fraction)
        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0.20, to=0.60, resolution=0.01, orient=tk.HORIZONTAL,
                                        variable=self.threshold_var, command=self.update_threshold_auto, length=160,
                                        bg="#FFF8DC", highlightthickness=0, font=("Arial", 8))
        self.threshold_scale.pack(pady=(0, 3))
        # VAT is relabelled when the slider is released (or after an arrow key), not while dragging
        self.threshold_scale.bind('<ButtonRelease-1>', self.apply_vat_fraction)
        self.threshold_scale.bind('<KeyRelease>', self.apply_vat_fraction)

        # SAT fat fraction slider (abdomen): a pixel outside the SAT boundary is SAT if its signal is at
        # least this share of the fat signal. The body outline and boundary always use abd_seg.FAT_FRACTION.
        tk.Label(self.threshold_frame, text="SAT Fat Fraction:", bg="#FFF8DC", font=("Arial", 9)).pack(pady=(3, 0))
        self.sat_var = tk.DoubleVar(value=self.sat_fraction)
        self.sat_scale = tk.Scale(self.threshold_frame, from_=0.20, to=0.60, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.sat_var, command=self.update_sat_fraction, length=160,
                                  bg="#FFF8DC", highlightthickness=0, font=("Arial", 8))
        self.sat_scale.pack(pady=(0, 3))
        self.sat_scale.bind('<ButtonRelease-1>', self.apply_vat_fraction)
        self.sat_scale.bind('<KeyRelease>', self.apply_vat_fraction)

        # IMAT contrast slider (thigh): an IMAT pixel stays IMAT if it lies at least this far from the
        # local muscle level (0) towards the local fat level (1), otherwise it becomes muscle.
        # Default per sequence: Dixon dixon_local_imat.CONTRAST, T1 TSE t1_seg.LOCAL_CONTRAST.
        self.imat_frame = tk.Frame(left_frame, bg="#E6F0FA", width=180, height=105)
        self.imat_frame.pack_propagate(False)
        tk.Label(self.imat_frame, text="IMAT Contrast:", bg="#E6F0FA", font=("Arial", 9)).pack(pady=(3, 0))
        self.imat_var = tk.DoubleVar(value=self.imat_contrast)
        self.imat_scale = tk.Scale(self.imat_frame, from_=0.10, to=0.90, resolution=0.01, orient=tk.HORIZONTAL,
                                   variable=self.imat_var, command=self.update_imat_contrast, length=160,
                                   bg="#E6F0FA", highlightthickness=0, font=("Arial", 8))
        self.imat_scale.pack(pady=(0, 3))
        # IMAT is relabelled when the slider is released (or after an arrow key), not while dragging
        self.imat_scale.bind('<ButtonRelease-1>', self.apply_imat_contrast)
        self.imat_scale.bind('<KeyRelease>', self.apply_imat_contrast)
        # "Advanced" unfolds the SAT Threshold and IMAT Radius sliders below (thigh_adv_frame)
        self.thigh_adv_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.imat_frame, text="Advanced", variable=self.thigh_adv_var, command=self.show_thigh_controls,
                       bg="#E6F0FA", activebackground="#E6F0FA", font=("Arial", 9)).pack()
        self.thigh_adv_frame = tk.Frame(left_frame, bg="#E6F0FA", width=180)

        # SAT threshold slider (thigh): fat above it, connected to the subcutaneous ring, is SAT.
        # Units and range per sequence (THIGH_SAT), set when a series is loaded. Changing it
        # runs Thigh Seg again (the SAT ring decides what is searched for IMAT and muscle).
        self.thigh_sat_label = tk.Label(self.thigh_adv_frame, text=THIGH_SAT['DIXON_F']['label'], bg="#E6F0FA", font=("Arial", 9))
        self.thigh_sat_label.pack(pady=(3, 0))
        self.thigh_sat_var = tk.DoubleVar(value=self.thigh_sat)
        cfg = THIGH_SAT['DIXON_F']
        self.thigh_sat_scale = tk.Scale(self.thigh_adv_frame, from_=cfg['from_'], to=cfg['to'], resolution=cfg['resolution'],
                                        orient=tk.HORIZONTAL, variable=self.thigh_sat_var, command=self.update_thigh_sat,
                                        length=160, bg="#E6F0FA", highlightthickness=0, font=("Arial", 8))
        self.thigh_sat_scale.pack(pady=(0, 3))
        self.thigh_sat_scale.bind('<ButtonRelease-1>', self.apply_thigh_sat)
        self.thigh_sat_scale.bind('<KeyRelease>', self.apply_thigh_sat)

        # IMAT radius slider (thigh, T1 only): neighbourhood (mm) of the local muscle level in the
        # IMAT check (t1_seg.LOCAL_RADIUS_MM). Changing it runs Thigh Seg again.
        self.thigh_radius_box = tk.Frame(self.thigh_adv_frame, bg="#E6F0FA")
        tk.Label(self.thigh_radius_box, text="IMAT Radius (mm, T1):", bg="#E6F0FA", font=("Arial", 9)).pack(pady=(3, 0))
        self.thigh_radius_var = tk.DoubleVar(value=self.thigh_radius)
        self.thigh_radius_scale = tk.Scale(self.thigh_radius_box, from_=4, to=30, resolution=1, orient=tk.HORIZONTAL,
                                           variable=self.thigh_radius_var, command=self.update_thigh_radius, length=160,
                                           bg="#E6F0FA", highlightthickness=0, font=("Arial", 8))
        self.thigh_radius_scale.pack(pady=(0, 3))
        self.thigh_radius_scale.bind('<ButtonRelease-1>', self.apply_thigh_sat)
        self.thigh_radius_scale.bind('<KeyRelease>', self.apply_thigh_sat)

        # SAT radius slider (thigh, T1 only): neighbourhood (mm) of the SAT local-contrast check
        # (t1_seg.check_sat); 0 = off (default). Changing it runs Thigh Seg again.
        tk.Label(self.thigh_radius_box, text="SAT Radius (mm, T1, 0 = off):", bg="#E6F0FA", font=("Arial", 9)).pack(pady=(3, 0))
        self.thigh_sat_radius_var = tk.DoubleVar(value=self.thigh_sat_radius)
        self.thigh_sat_radius_scale = tk.Scale(self.thigh_radius_box, from_=0, to=30, resolution=1, orient=tk.HORIZONTAL,
                                               variable=self.thigh_sat_radius_var, command=self.update_thigh_sat_radius,
                                               length=160, bg="#E6F0FA", highlightthickness=0, font=("Arial", 8))
        self.thigh_sat_radius_scale.pack(pady=(0, 3))
        self.thigh_sat_radius_scale.bind('<ButtonRelease-1>', self.apply_thigh_sat)
        self.thigh_sat_radius_scale.bind('<KeyRelease>', self.apply_thigh_sat)

        # Right frame
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        right_frame.grid_columnconfigure(0, weight=3)
        right_frame.grid_columnconfigure(1, weight=2)
        right_frame.grid_rowconfigure(0, weight=1)
        
        top_frame = tk.Frame(right_frame)
        top_frame.grid(row=0, column=0, sticky="nsew")
        
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.grid(row=0, column=1, sticky="nsew")
        
        # Long axis view (right column, above the plot)
        self.fig_long_2d = plt.figure(figsize=(4, 4))
        self.ax_long_2d = self.fig_long_2d.add_subplot(111)
        canvas_long_2d = FigureCanvasTkAgg(self.fig_long_2d, bottom_frame)
        canvas_long_2d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax_long_2d.spines['top'].set_visible(False)
        self.ax_long_2d.spines['right'].set_visible(False)
        self.ax_long_2d.spines['bottom'].set_visible(False)
        self.ax_long_2d.spines['left'].set_visible(False)
        self.ax_long_2d.xaxis.set_ticks([])
        self.ax_long_2d.yaxis.set_ticks([])
        
        self.long_slider = tk.Scale(bottom_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Slider")
        self.long_slider.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.long_slider.bind("<ButtonRelease-1>", self.update_long_slice)

        # Slice view (short axis; large left column)
        self.fig_slice = plt.figure(figsize=(4, 4))
        self.ax_slice = self.fig_slice.add_subplot(111)
        canvas_slice = FigureCanvasTkAgg(self.fig_slice, top_frame)
        canvas_slice.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Three button slider (slice and crop range): below the short-axis view
        self.twoSlider = ThreeButtonSlider(top_frame, 0, 10, 0, 5, 10, 200, 20, parent=self)
        self.twoSlider.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=(10, 10))

        # Fat plot (right column, bottom)
        self.fig_plot = plt.figure(figsize=(4, 4))
        self.fat_plot = self.fig_plot.add_subplot(111)
        canvas_plot = FigureCanvasTkAgg(self.fig_plot, bottom_frame)
        canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.max_crop = 0
        self.min_crop = 0
        self.current = 0
        self.elevation, self.azimuth = 30, -60
        self.over_image = 0
        self.current_mode = "clear"
        self.segmentation = None
        self.line_drawing_active = False
        self.start_point = None
        self.end_point = None
        self.permanent_lines = []
        
        # Initialize button display for SAT mode
        self.update_button_image()

        # Automatically click switch button once and then lock it
        self.update_button_image()  # Auto-click once
        # self.switch_button.config(state=tk.DISABLED)  # Lock the button

        # Start periodic update
        self.update_two_values()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the segmentation worker processes in the background, so the first run doesn't wait for them
        parallel.warm_up()
    
    def on_closing(self):
        # Cancel the after callback
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if self.busy_after_id is not None:
            self.root.after_cancel(self.busy_after_id)
        parallel.shutdown()
        plt.close('all')
        self.root.quit()
        self.root.destroy()
    
    def clear_views(self):
        """Drop the loaded series, segmentation and drawn lines, and blank every view.

        Used when the mode changes: the images and labels of one mode (abdomen SAT/VAT,
        thigh SAT/IMAT/muscle) do not belong in the other.
        """
        self.dicom_short_pixels = self.dicom_short_info = None
        self.dicom_long_pixels = self.dicom_long_info = None
        if hasattr(self, 'image_stack'):
            del self.image_stack
        self.short_sequence = None
        self.segmentation = None
        self.abd_state = self.abd_labels = None
        self.thigh_state = self.thigh_labels = None
        self.ai_seg_active = False
        self.seg_transposed = None
        self.__dict__.pop('affine', None)
        self.permanent_lines = []
        self.start_point = self.end_point = None
        self.line_drawing_active = False
        if self.flip_axes:
            self.flip_axes = False
            self.flip_button.config(text="Flip Seg X/Y", bg="#FF8C00")
        popup = getattr(self, 'image_popup', None)
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        self.image_popup = None
        for ax in (self.ax_long_2d, self.ax_slice, self.fat_plot):
            ax.clear()
        self.ax_long_2d.set_axis_off()
        self.ax_slice.set_axis_off()
        for fig in (self.fig_long_2d, self.fig_slice, self.fig_plot):
            fig.canvas.draw()

    def update_button_image(self, confirm=True):
        """Switch between abdomen and thigh mode; the views of the old mode are cleared.

        confirm: ask first when a segmentation would be lost (False when loading a series
        switches the mode itself).
        """
        if self.click_state in (1, 2) and getattr(self, 'fig_slice', None) is not None:
            if confirm and getattr(self, 'segmentation', None) is not None:
                if not messagebox.askyesno("Switch mode",
                                           "Switching mode clears the loaded images and the segmentation.\n"
                                           "Save the segmentation first if you need it. Switch now?"):
                    return
            self.clear_views()
        self.load_long_button.pack(pady=5)
        self.load_short_button.pack(pady=5)
        self.load_seg_button.pack(pady=5)
        self.flip_button.pack(pady=5)
        self.show_image_button.pack(pady=5)
        self.save_image_button.pack(pady=5)
        
        if self.click_state == 0:
            self.switch_button.config(image=self.sat_img)
            self.click_state = 1
            self.auto_seg_button.pack(pady=5)
            self.threshold_frame.pack(padx=5, pady=5, after=self.left_bottom_frame)   # VAT Fat Fraction: abdomen only
            self.imat_frame.pack_forget()
            self.thigh_adv_frame.pack_forget()
        elif self.click_state == 1:
            self.switch_button.config(image=self.thigh_img)
            self.click_state = 2
            self.draw_line_button.pack(pady=5)
            self.thigh_button.pack(pady=5)
            self.ai_seg_button.pack(pady=5)
            self.combined_seg_button.pack(pady=5)
            self.swap_xy_button.pack(pady=5)
            self.auto_seg_button.pack_forget()
            self.threshold_frame.pack_forget()   # VAT Fat Fraction is not used in thigh mode
            self.imat_frame.pack(padx=5, pady=5, after=self.left_bottom_frame)   # IMAT Contrast: thigh only
            self.show_thigh_controls()
        else:
            self.switch_button.config(image=self.sat_img)
            self.click_state = 1
            self.draw_line_button.pack_forget()
            self.thigh_button.pack_forget()
            self.ai_seg_button.pack_forget()
            self.combined_seg_button.pack_forget()
            self.swap_xy_button.pack_forget()
            self.auto_seg_button.pack(pady=5)
            self.threshold_frame.pack(padx=5, pady=5, after=self.left_bottom_frame)   # VAT Fat Fraction: abdomen only
            self.imat_frame.pack_forget()
            self.thigh_adv_frame.pack_forget()

    def show_thigh_controls(self):
        """Thigh mode: fold / unfold the advanced sliders; IMAT Radius only for T1 TSE."""
        if self.click_state != 2 or not self.thigh_adv_var.get():
            self.thigh_adv_frame.pack_forget()
            return
        self.thigh_adv_frame.pack(padx=5, pady=5, after=self.imat_frame)
        if self.short_sequence == 'T1_TSE':
            self.thigh_radius_box.pack()
        else:
            self.thigh_radius_box.pack_forget()

    def flip_xy_axes(self):
        """Flip X and Y axes of the segmentation only."""
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            messagebox.showwarning("Warning", "No segmentation loaded.")
            return

        self.flip_axes = not self.flip_axes

        # Apply flip to segmentation only
        self.segmentation = np.transpose(self.segmentation, (1, 0, 2))

        # Update button appearance
        if self.flip_axes:
            self.flip_button.config(text="Unflip Seg X/Y", bg="#32CD32")
        else:
            self.flip_button.config(text="Flip Seg X/Y", bg="#FF8C00")

        # Update display
        self.update_image_slice()
        self.update_fat_plot()
        messagebox.showinfo("Info", f"Segmentation X/Y axes {'flipped' if self.flip_axes else 'unflipped'}")

    def get_display_slice(self, slice_data):
        """Get slice data with flip transformation applied if needed."""
        if self.flip_axes:
            return slice_data.T
        return slice_data

    def seg_display_slice(self, index, image_shape):
        """Segmentation slice `index` in the orientation of the image (shape image_shape).

        Uses the known layout (seg_transposed); a loaded or flipped segmentation whose shape
        still differs from the image is transposed (square images cannot be told apart).
        """
        s = self.segmentation[:, :, index]
        if self.seg_transposed:
            s = s.T
        if s.shape != tuple(image_shape):
            s = s.T
        return s

    def to_seg_orientation(self, mask):
        """A mask in image orientation, in the layout of the segmentation array (inverse of seg_display_slice)."""
        if self.seg_transposed:
            mask = mask.T
        if mask.shape != self.segmentation.shape[:2]:
            mask = mask.T
        return mask

    def choose_series(self, options):
        """Ask which acquisition to load when several in the folder match (parseDicomFolder's choose_series).

        Returns the index of the chosen option, or None if the dialog is closed. The latest
        series (highest number) is preselected.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose series")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        tk.Label(dialog, text="Several series in this folder match. Load which one?",
                 font=("Arial", 10)).pack(padx=15, pady=(12, 6))

        def fmt_time(t):
            t = str(t).split('.')[0]
            return f"{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t) >= 6 else t

        choice = tk.IntVar(value=len(options) - 1)
        for i, o in enumerate(options):
            tk.Radiobutton(dialog, variable=choice, value=i, anchor=tk.W, justify=tk.LEFT,
                           text=f"#{o['number']}  {o['name']}  -  {o['slices']} slices, {o['shape'][0]} x {o['shape'][1]}"
                                f"  (time {fmt_time(o['time'])})").pack(fill=tk.X, padx=15)
        result = {}
        buttons = tk.Frame(dialog)
        buttons.pack(pady=10)

        def ok():
            result['index'] = choice.get()
            dialog.destroy()

        tk.Button(buttons, text="Load", width=10, command=ok).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="Cancel", width=10, command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        dialog.grab_set()
        dialog.wait_window()
        return result.get('index')

    def load_short_files(self):
        folder = filedialog.askdirectory(title="Select DICOM Directory")
        if not folder:
            return
        try:
            # Thigh mode: Dixon fat series (exact name), else T1 TSE (name contains "t1_tse_tra");
            # abdomen mode: any series ending with "_F". self.short_sequence picks the thigh method.
            if self.click_state == 2:
                try:
                    self.dicom_short_pixels, self.dicom_short_info = parseDicomFolder(folder, '6pt_DIXON_VIBE_F', choose_series=self.choose_series)
                    self.short_sequence = 'DIXON_F'
                except ValueError:
                    self.dicom_short_pixels, self.dicom_short_info = parseDicomFolder(folder, seriesContains='t1_tse_tra', choose_series=self.choose_series)
                    self.short_sequence = 'T1_TSE'
            else:
                try:
                    self.dicom_short_pixels, self.dicom_short_info = parseDicomFolder(folder, seriesSuffix='_F', choose_series=self.choose_series)
                    self.short_sequence = 'DIXON_F'
                except ValueError as no_f_series:
                    # T1 TSE is a thigh sequence: offer to switch to thigh mode instead of failing
                    try:
                        pixels, info = parseDicomFolder(folder, seriesContains='t1_tse_tra', choose_series=self.choose_series)
                    except ValueError:
                        raise no_f_series
                    if not messagebox.askyesno("Thigh sequence",
                                               f"This folder has no '_F' series, but has the thigh sequence '{info[0][1]}'.\n"
                                               "Switch to thigh mode and load it?"):
                        return
                    self.update_button_image(confirm=False)  # abdomen -> thigh mode (clears the old views)
                    self.dicom_short_pixels, self.dicom_short_info = pixels, info
                    self.short_sequence = 'T1_TSE'
            self.image_stack = np.array(self.dicom_short_pixels)
            self.abd_state = self.abd_labels = None   # boundary of a previous stack no longer applies
            self.thigh_state = self.thigh_labels = None
            # IMAT Contrast starts at the default of the loaded sequence
            self.imat_contrast = t1_seg.LOCAL_CONTRAST if self.short_sequence == 'T1_TSE' else dixon_local_imat.CONTRAST
            self.imat_var.set(self.imat_contrast)
            # SAT Threshold: units, range and default of the loaded sequence
            cfg = THIGH_SAT['T1_TSE' if self.short_sequence == 'T1_TSE' else 'DIXON_F']
            self.thigh_sat = cfg['default']
            self.thigh_sat_scale.config(from_=cfg['from_'], to=cfg['to'], resolution=cfg['resolution'])
            self.thigh_sat_label.config(text=cfg['label'])
            self.thigh_sat_var.set(self.thigh_sat)
            self.thigh_radius = int(t1_seg.LOCAL_RADIUS_MM)
            self.thigh_radius_var.set(self.thigh_radius)
            self.thigh_sat_radius = int(t1_seg.SAT_RADIUS_MM)
            self.thigh_sat_radius_var.set(self.thigh_sat_radius)
            self.show_thigh_controls()   # IMAT Radius only for T1
            # The segmentation belongs to the previous stack: drop it, and the plot drawn from it
            self.segmentation = None
            self.ai_seg_active = False
            self.seg_transposed = None
            self.__dict__.pop('affine', None)   # affine of a seg file loaded for the previous stack
            if self.flip_axes:
                self.flip_axes = False
                self.flip_button.config(text="Flip Seg X/Y", bg="#FF8C00")
            self.fat_plot.clear()
            self.fat_plot.figure.canvas.draw()
            # print('image stack shape: ', self.image_stack.shape)
            self.min_crop = 0
            self.max_crop = self.image_stack.shape[0] - 1
            self.current = int(self.max_crop / 2)
            self.twoSlider.set_values2(self.min_crop, self.max_crop)
            self.twoSlider.set_values(self.min_crop, self.current, self.max_crop)
            self.z_positions = [info[6] for info in self.dicom_short_info]
            self.image_stack = pipeline.clipped_stack(self.dicom_short_pixels)   # SAT/AVT Seg runs on this
            self.update_image_slice()
        except SeriesChoiceCancelled:
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load short axis slices: {str(e)}")
    
    def load_long_files(self):
        folder = filedialog.askdirectory(title="Select DICOM Directory")
        if not folder:
            return
        try:
            self.dicom_long_pixels, self.dicom_long_info = parseDicomFolder(folder, choose_series=self.choose_series) #, 'localizer_cor')
            self.dicom_long_pixels = np.array(self.dicom_long_pixels)
            self.long_min = 0
            self.long_max = self.dicom_long_pixels.shape[0]
            self.long_slider.config(from_=0, to=self.long_max - 1)
            self.update_long_slice()
        except SeriesChoiceCancelled:
            return
        except Exception as e:
            messagebox.showerror("Error", f"Will display long axis slices after loading short axis slices") #: {str(e)}")
    
    def load_segmentation_files(self):
        filename = filedialog.askopenfilename(title="Select Segmentation File", filetypes=[("NII.GZ files", "*.nii.gz")])
        if not filename:
            return
        try:
            self.affine = nib.load(filename).affine
            self.segmentation = nib.load(filename).get_fdata()
            self.ai_seg_active = False
            self.seg_transposed = None   # layout of a file is unknown
            self.update_image_slice()
            self.update_fat_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load segmentation file: {str(e)}")
    
    def dicom_affine(self):
        """NIfTI affine of an array laid out like the short-axis images (see dicom_io.dicom_affine)."""
        return dicom_affine(self.dicom_short_info)

    def save_segmentation(self):
        if getattr(self, 'segmentation', None) is None:
            messagebox.showwarning("Warning", "No segmentation to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".nii.gz",
            filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")],
            title="Save Segmentation As"
        )
        if file_path:
            if not file_path.endswith(".nii.gz"):
                file_path += ".nii.gz"
            try:
                images = getattr(self, 'image_stack', None)
                if images is not None and self.segmentation.shape[2] == len(images):
                    # Save what is shown: every slice in the layout of the images (rows, cols, slices),
                    # whichever method made it, with the geometry of the loaded DICOM series
                    segmentation_data = np.stack([self.seg_display_slice(i, images[i].shape) for i in range(len(images))],
                                                 axis=-1).astype(np.int16)
                    affine = self.dicom_affine()
                else:
                    # No matching images loaded: keep the array and the affine it was loaded with
                    segmentation_data = self.segmentation.astype(np.int16)
                    affine = getattr(self, 'affine', np.eye(4))
                seg = nib.Nifti1Image(segmentation_data, affine)
                nib.save(seg, file_path)
                messagebox.showinfo("Success", f"Segmentation saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save segmentation: {str(e)}")
    
    def update_threshold_auto(self, value):
        """Keep the VAT fat fraction in step with the slider (VAT is relabelled on release)."""
        self.fat_fraction = round(float(value), 2)

    def update_sat_fraction(self, value):
        """Keep the SAT fat fraction in step with the slider (SAT is relabelled on release)."""
        self.sat_fraction = round(float(value), 2)

    def busy_widgets(self):
        """Controls that are disabled while a segmentation runs."""
        return [self.switch_button, self.load_short_button, self.load_long_button, self.load_seg_button,
                self.flip_button, self.auto_seg_button, self.show_image_button, self.save_image_button,
                self.draw_line_button, self.thigh_button, self.ai_seg_button, self.swap_xy_button,
                self.combined_seg_button, self.threshold_scale, self.sat_scale, self.imat_scale, self.thigh_sat_scale, self.thigh_radius_scale, self.thigh_sat_radius_scale]

    def run_busy(self, title, work, done):
        """Run work() in a background thread while a popup shows a running timer, then done(result).

        work() must not touch Tk widgets; done() runs on the Tk thread and updates the display.
        """
        if self.busy:
            return
        self.busy = True
        for w in self.busy_widgets():
            w.config(state=tk.DISABLED)

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.resizable(False, False)
        popup.transient(self.root)                      # stays on top of the main window
        popup.protocol("WM_DELETE_WINDOW", lambda: None)  # cannot be closed while running
        label = tk.Label(popup, text=f"{title}...  0.0 s", font=("Arial", 12, "bold"), width=28)
        label.pack(padx=20, pady=(15, 8))
        bar = ttk.Progressbar(popup, mode='indeterminate', length=260)
        bar.pack(padx=20, pady=(0, 15))
        bar.start(15)
        # Centre the popup over the main window
        popup.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - popup.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        start = time.perf_counter()
        box = {}

        def target():
            try:
                box['result'] = work()
            except Exception as e:
                traceback.print_exc()
                box['error'] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        def poll():
            elapsed = time.perf_counter() - start
            if thread.is_alive():
                label.config(text=f"{title}...  {elapsed:.1f} s")
                self.busy_after_id = self.root.after(100, poll)
                return
            self.busy_after_id = None
            bar.stop()
            for w in self.busy_widgets():
                w.config(state=tk.NORMAL)
            self.busy = False
            if 'error' in box:
                popup.destroy()
                messagebox.showerror("Error", f"{title} failed: {box['error']}")
                return
            print(f"{title} done in {elapsed:.1f} s")
            try:
                done(box['result'])
            except Exception as e:
                traceback.print_exc()
                popup.destroy()
                messagebox.showerror("Error", f"{title}: failed to show the result: {e}")
                return
            # Show the final time briefly, then close (or close straight away on click)
            bar.pack_forget()
            label.config(text=f"{title} done in {elapsed:.1f} s", fg="darkgreen")
            popup.protocol("WM_DELETE_WINDOW", popup.destroy)
            popup.bind("<Button-1>", lambda e: popup.destroy())
            self.root.after(1500, lambda: popup.winfo_exists() and popup.destroy())

        poll()

    def auto_segment(self):
        if not hasattr(self, 'image_stack'):
            messagebox.showwarning("Warning", "No image stack loaded.")
            return
        pixel_spacing = float(self.dicom_short_info[0][4][0])
        stack, fraction, sat_fraction = self.image_stack, self.fat_fraction, self.sat_fraction

        def work():
            return segment_abdomen_stack(stack, pixel_spacing=pixel_spacing, vat_fraction=fraction,
                                         sat_fraction=sat_fraction, return_state=True)

        def done(result):
            self.segmentation, self.abd_state = result
            self.abd_labels = self.segmentation.copy()
            self.abd_state_fraction = fraction
            self.abd_state_sat_fraction = sat_fraction
            self.ai_seg_active = False
            self.seg_transposed = False
            self.update_image_slice()
            self.update_fat_plot()

        self.run_busy("SAT/AVT Seg", work, done)

    def apply_vat_fraction(self, event=None):
        """Relabel SAT / VAT with the current SAT and VAT Fat Fractions, reusing the last SAT/AVT Seg boundary."""
        if self.click_state != 1 or self.abd_state is None or not hasattr(self, 'image_stack'):
            return
        if (self.fat_fraction == getattr(self, 'abd_state_fraction', None)
                and self.sat_fraction == getattr(self, 'abd_state_sat_fraction', None)):
            return
        current = getattr(self, 'segmentation', None)
        if current is not None and self.abd_labels is not None and not np.array_equal(current, self.abd_labels):
            if not messagebox.askyesno("Fat Fraction",
                                       "The segmentation was edited or loaded since SAT/AVT Seg.\n"
                                       "Relabel SAT/VAT with the new values and discard those changes?"):
                return
        stack, state, fraction, sat_fraction = self.image_stack, self.abd_state, self.fat_fraction, self.sat_fraction

        def done(labels):
            self.segmentation = labels
            self.seg_transposed = False
            self.abd_labels = self.segmentation.copy()
            self.abd_state_fraction = fraction
            self.abd_state_sat_fraction = sat_fraction
            print(f"Relabelled with SAT fat fraction {sat_fraction}, VAT fat fraction {fraction}")
            self.update_image_slice()
            self.update_fat_plot()

        self.run_busy("SAT/VAT relabel", lambda: relabel_vat(stack, state, fraction, sat_fraction), done)
    
    def update_imat_contrast(self, value):
        """Keep the IMAT contrast in step with the slider (IMAT is relabelled on release)."""
        self.imat_contrast = round(float(value), 2)

    def apply_imat_contrast(self, event=None):
        """Relabel IMAT / muscle with the current IMAT Contrast, reusing the last Thigh Seg."""
        if self.click_state != 2 or self.thigh_state is None:
            return
        if self.imat_contrast == getattr(self, 'thigh_state_contrast', None):
            return
        current = getattr(self, 'segmentation', None)
        if current is not None and self.thigh_labels is not None and not np.array_equal(current, self.thigh_labels):
            if not messagebox.askyesno("IMAT Contrast",
                                       "The segmentation was edited or loaded since Thigh Seg.\n"
                                       "Relabel IMAT with the new value and discard those changes?"):
                return
        state, contrast = self.thigh_state, self.imat_contrast
        sliceId = self.twoSlider.get_values()[1]

        def done(labels):
            self.segmentation = np.array(labels).transpose(1, 2, 0)
            self.thigh_labels = self.segmentation.copy()
            self.thigh_state_contrast = contrast
            self.ai_seg_active = False
            self.seg_transposed = False
            print(f"IMAT relabelled with IMAT contrast {contrast}")
            self.update_image_slice()
            self.update_short_slice()
            self.update_popup_image(sliceId)

        self.run_busy("IMAT relabel", lambda: t1_seg.relabel_imat(state, contrast), done)

    def update_thigh_sat(self, value):
        """Keep the thigh SAT threshold in step with the slider (Thigh Seg runs again on release)."""
        cfg = THIGH_SAT['T1_TSE' if self.short_sequence == 'T1_TSE' else 'DIXON_F']
        self.thigh_sat = round(float(value), 2) if cfg['resolution'] < 1 else int(round(float(value)))

    def update_thigh_radius(self, value):
        """Keep the T1 IMAT radius in step with the slider (Thigh Seg runs again on release)."""
        self.thigh_radius = int(round(float(value)))

    def update_thigh_sat_radius(self, value):
        """Keep the T1 SAT radius in step with the slider (Thigh Seg runs again on release)."""
        self.thigh_sat_radius = int(round(float(value)))

    def apply_thigh_sat(self, event=None):
        """Run Thigh Seg again with the new SAT threshold / IMAT radius / SAT radius, if there is a Thigh Seg to update."""
        if self.click_state != 2 or self.thigh_state is None:
            return
        if (self.thigh_sat == getattr(self, 'thigh_state_sat', None)
                and self.thigh_radius == getattr(self, 'thigh_state_radius', None)
                and self.thigh_sat_radius == getattr(self, 'thigh_state_sat_radius', None)):
            return
        current = getattr(self, 'segmentation', None)
        if current is not None and self.thigh_labels is not None and not np.array_equal(current, self.thigh_labels):
            if not messagebox.askyesno("Thigh settings",
                                       "The segmentation was edited or loaded since Thigh Seg.\n"
                                       "Run Thigh Seg with the new value and discard those changes?"):
                return
        self.thigh_mode()

    def thigh_mode(self):
        if self.click_state != 2:
            messagebox.showwarning("Mode Incorrect", "Thigh segmentation is only available in Thigh Seg mode.")
            return

        sliceId = self.twoSlider.get_values()[1]

        if self.dicom_short_info is None or self.dicom_short_pixels is None:
            # print("No short axis slices loaded.")
            return
        pixels, info, sequence = self.dicom_short_pixels, self.dicom_short_info, self.short_sequence
        contrast, sat, radius, sat_radius = self.imat_contrast, self.thigh_sat, self.thigh_radius, self.thigh_sat_radius

        def work():
            # (slices, H, W): 1 SAT, 2 IMAT, 3 muscle; separate method per sequence (pipeline.py, the
            # same steps batch_seg.py runs). The state lets the IMAT Contrast slider relabel IMAT /
            # muscle without segmenting again.
            return pipeline.segment_thigh(pixels, info, sequence, sat_thr=sat, contrast=contrast, radius_mm=radius,
                                          sat_radius_mm=sat_radius, return_state=True)

        def done(result):
            fatSeg, self.thigh_state = result
            self.segmentation = np.array(fatSeg).transpose(1, 2, 0)
            self.thigh_labels = self.segmentation.copy()
            self.thigh_state_contrast = contrast
            self.thigh_state_sat = sat
            self.thigh_state_radius = radius
            self.thigh_state_sat_radius = sat_radius
            print(f"Thigh Seg with SAT threshold {sat}, IMAT contrast {contrast}"
                  + (f", IMAT radius {radius} mm, SAT radius {sat_radius or 'off'}" + (' mm' if sat_radius else '')
                     if sequence == 'T1_TSE' else ''))
            self.ai_seg_active = False
            self.seg_transposed = False

            self.update_image_slice()
            self.update_short_slice()
            self.update_popup_image(sliceId)

        self.run_busy("Thigh Seg", work, done)

    def ai_seg_mode(self):
        if self.click_state != 2:
            messagebox.showwarning("Mode Incorrect", "AI segmentation is only available in Thigh Seg mode.")
            return

        if self.dicom_short_info is None or self.dicom_short_pixels is None:
            messagebox.showwarning("Warning", "No short axis slices loaded.")
            return

        if self.short_sequence == 'T1_TSE':
            messagebox.showwarning("Sequence", "The AI model was trained on Dixon fat (_F) images only. Use Thigh Seg for t1_tse_tra.")
            return

        IMG_HEIGHT, IMG_WIDTH = 256, 256

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        finetuned = os.path.join(base_dir, 'unet_thighfat_finetuned.keras')
        model_path = finetuned if os.path.exists(finetuned) else os.path.join(base_dir, 'unet_thighfat_segmentation_model_best_loss.keras')
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return

        sliceId = self.twoSlider.get_values()[1]
        pixels = self.dicom_short_pixels

        def work():
            # Imported here, not at the top: it takes 10-20 s and only AI Seg needs it (and the
            # parallel workers re-import this file on Windows)
            import tensorflow as tf
            # Load the model once and reuse it; reload only if a different file is picked
            # (e.g. the fine-tuned model was copied in after the first run)
            if self.ai_model is None or self.ai_model_path != model_path:
                self.ai_model = tf.keras.models.load_model(model_path)
                self.ai_model_path = model_path
            # Per slice (W, H) masks, foreground (SAT, IMAT, muscle) merged into label 1
            return pipeline.predict_ai(self.ai_model, pixels, size=(IMG_HEIGHT, IMG_WIDTH))

        def done(fatSeg):
            self.segmentation = np.array(fatSeg).transpose(1, 2, 0)
            self.ai_seg_active = True
            self.seg_transposed = True   # masks were predicted on the transposed slices

            self.update_image_slice()
            self.update_short_slice()
            self.update_popup_image(sliceId)

        self.run_busy("AI Seg", work, done)

    def swap_xy_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            messagebox.showwarning("Warning", "No segmentation loaded.")
            return

        # Swap the x and y axes of every slice (a transpose). It used to resize the result back to
        # the old shape, which stretched the labels of non-square slices. A non-square result
        # that no longer fits the images is shown transposed back (seg_display_slice), like Flip Seg X/Y.
        self.segmentation = np.ascontiguousarray(np.transpose(self.segmentation, (1, 0, 2)))

        sliceId = self.twoSlider.get_values()[1]
        self.update_image_slice()
        self.update_short_slice()
        self.update_popup_image(sliceId)

    def combined_seg_mode(self):
        if self.click_state != 2:
            messagebox.showwarning("Mode Incorrect", "Combined segmentation is only available in Thigh Seg mode.")
            return

        if self.short_sequence == 'T1_TSE':
            messagebox.showwarning("Sequence", "Combined uses the AI model, which was trained on Dixon fat (_F) images only. Use Thigh Seg for t1_tse_tra.")
            return

        if getattr(self, 'segmentation', None) is None or not self.ai_seg_active:
            messagebox.showwarning("Warning", "Combined needs an AI segmentation. Run AI Seg first.")
            return

        if self.dicom_short_info is None or self.dicom_short_pixels is None:
            messagebox.showwarning("Warning", "No short axis slices loaded.")
            return

        if self.segmentation.shape[2] != len(self.dicom_short_pixels):
            messagebox.showwarning("Warning", "The segmentation does not match the loaded slices. Run AI Seg again.")
            return

        sliceId = self.twoSlider.get_values()[1]

        pixels, info = self.dicom_short_pixels, self.dicom_short_info
        segmentation = self.segmentation.copy()
        contrast, sat = self.imat_contrast, self.thigh_sat
        to_seg = self.to_seg_orientation   # depends only on seg_transposed and the array shape

        def work():
            # Dixon Thigh Seg (seg.py + dixon_local_imat.py, as in pipeline.py)
            conventional = pipeline.segment_thigh(pixels, info, 'DIXON_F', sat_thr=sat, contrast=contrast)
            for s in range(segmentation.shape[2]):
                green_mask = (segmentation[:, :, s] > 0).astype(np.uint8)

                # Conventional results are in image orientation; bring them to the segmentation's layout
                seg2_t = to_seg(conventional[s] == 2)
                muscle_t = to_seg(conventional[s] == 3)

                combined = green_mask.copy()  # keep green (1) as base
                combined[seg2_t & (green_mask > 0)] = 2   # red (IMAT)
                combined[muscle_t & (green_mask > 0)] = 3  # blue (muscle)

                segmentation[:, :, s] = combined
            return segmentation

        def done(result):
            self.segmentation = result   # same layout as the AI segmentation; Combined can be run again

            self.update_image_slice()
            self.update_short_slice()
            self.update_popup_image(sliceId)

        self.run_busy("Combined", work, done)

    def activate_line_drawing(self):
        if hasattr(self, 'temporary_line'):
            try:
                self.temporary_line.remove()
            except:
                pass
            del self.temporary_line
        
        self.line_drawing_active = not self.line_drawing_active
        if self.line_drawing_active:
            self.start_point = None
            self.end_point = None
            self.permanent_lines = []
            self.draw_line_button.config(bg="#4CAF50", fg="white")
            self.cid_press = self.fig_long_2d.canvas.mpl_connect('button_press_event', self.start_line)
            self.cid_motion = self.fig_long_2d.canvas.mpl_connect('motion_notify_event', self.draw_temporary_line)
            self.cid_release = self.fig_long_2d.canvas.mpl_connect('button_release_event', self.finalize_line)
        else:
            self.draw_line_button.config(bg="lightgray", fg="black")
            if hasattr(self, 'cid_press'):
                self.fig_long_2d.canvas.mpl_disconnect(self.cid_press)
                self.fig_long_2d.canvas.mpl_disconnect(self.cid_motion)
                self.fig_long_2d.canvas.mpl_disconnect(self.cid_release)
            if hasattr(self, 'temporary_line'):
                try:
                    self.temporary_line.remove()
                except:
                    pass
                del self.temporary_line
        self.fig_long_2d.canvas.draw()
    
    def start_line(self, event):
        if event.inaxes == self.ax_long_2d and self.line_drawing_active:
            #if not hasattr(self, 'temporary_line'):
            self.start_point = (event.xdata, event.ydata)
    
    def draw_temporary_line(self, event):
        if event.inaxes == self.ax_long_2d and self.line_drawing_active and self.start_point:
            if hasattr(self, 'temporary_line'):
                try:
                    self.temporary_line.remove()
                except:
                    pass
            self.end_point = (event.xdata, event.ydata)
            self.temporary_line = self.ax_long_2d.plot(
                [self.start_point[0], self.end_point[0]],
                [self.start_point[1], self.end_point[1]],
                color='yellow', linewidth=2, alpha=0.5
            )[0]
            self.fig_long_2d.canvas.draw()
    
    def finalize_line(self, event):
        if event.inaxes == self.ax_long_2d and self.line_drawing_active and self.start_point:
            self.end_point = (event.xdata, event.ydata)
            current_long_info = self.dicom_long_info[int(self.long_slider.get())]
            current_long_pixels = self.dicom_long_pixels[int(self.long_slider.get())]
            middle_point = (np.array(self.start_point) + np.array(self.end_point)) / 2
            # x (column index) runs along the row direction cosines (IOP[:3]) in steps of the column
            # spacing PixelSpacing[1]; y (row index) along IOP[3:] in steps of the row spacing PixelSpacing[0]
            middle_point_3d = (np.array(current_long_info[3], float)
                               + middle_point[0] * np.array(current_long_info[2][:3], float) * float(current_long_info[4][1])
                               + middle_point[1] * np.array(current_long_info[2][3:], float) * float(current_long_info[4][0]))
            short_normal = np.cross(self.dicom_short_info[0][2][:3], self.dicom_short_info[0][2][3:])
            short_normal = short_normal / np.linalg.norm(short_normal)
            short_normal_position = [np.abs(np.dot(k[3], short_normal) - np.dot(middle_point_3d, short_normal)) for k in self.dicom_short_info]
            selected_index = np.argmin(short_normal_position)
            self.twoSlider.set_values(self.min_crop, selected_index, self.max_crop)
            self.permanent_lines = [(self.start_point, self.end_point)]
            if hasattr(self, 'temporary_line'):
                try:
                    self.temporary_line.remove()
                except:
                    # pass
                    del self.temporary_line
            self.update_long_2d()
            self.update_image_slice()
            self.start_point = None
            self.end_point = None
    
    def update_long_2d(self):
        def getCrossList(value):
            current_slice = int(value)
            current_origin = self.dicom_short_info[current_slice][2]
            current_position = self.dicom_short_info[current_slice][3]
            current_xdir = np.array(current_origin[:3])
            current_ydir = np.array(current_origin[3:])
            long_index = int(self.long_slider.get())
            long_origin = self.dicom_long_info[long_index][2]
            long_position = self.dicom_long_info[long_index][3]
            long_xdir = np.array(long_origin[:3])
            long_ydir = np.array(long_origin[3:])
            end0 = long_position
            end1 = long_position + long_ydir * self.dicom_long_pixels[long_index].shape[0] * self.dicom_long_info[long_index][4][0]
            end2 = long_position + long_xdir * self.dicom_long_pixels[long_index].shape[1] * self.dicom_long_info[long_index][4][1]
            end3 = end2 + long_ydir * self.dicom_long_pixels[long_index].shape[0] * self.dicom_long_info[long_index][4][0]
            pt0y = line_plane_intersection(current_xdir, current_ydir, current_position, end0, end1)
            pt0x = 0
            pt0 = [pt0x / float(self.dicom_long_info[long_index][4][1]), pt0y / float(self.dicom_long_info[long_index][4][0])]
            pt1y = line_plane_intersection(current_xdir, current_ydir, current_position, end2, end3)
            pt1x = self.dicom_long_pixels[long_index].shape[1] * self.dicom_long_info[long_index][4][1]
            pt1 = [pt1x / float(self.dicom_long_info[long_index][4][1]), pt1y / float(self.dicom_long_info[long_index][4][0])]
            return pt0, pt1
        
        if self.dicom_long_pixels is None:
            return
        
        min_crop, value, max_crop = self.twoSlider.get_values()
        pt0, pt1 = getCrossList(value)
        red0, red1 = getCrossList(max_crop)
        green0, green1 = getCrossList(min_crop)
        
        self.ax_long_2d.clear()
        self.ax_long_2d.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'w-')
        self.ax_long_2d.plot([red0[0], red1[0]], [red0[1], red1[1]], 'r-')
        self.ax_long_2d.plot([green0[0], green1[0]], [green0[1], green1[1]], 'b-')
        self.ax_long_2d.imshow(self.dicom_long_pixels[int(self.long_slider.get())], cmap='gray', origin='upper')
        self.ax_long_2d.set_title("Long Axis View")
        self.ax_long_2d.set_axis_off()
        self.ax_long_2d.set_xlim(0, self.dicom_long_pixels[0].shape[1])
        self.ax_long_2d.set_ylim(0, self.dicom_long_pixels[0].shape[0])
        self.ax_long_2d.invert_yaxis()
        self.ax_long_2d.invert_xaxis()
        
        for start, end in self.permanent_lines:
            self.ax_long_2d.plot(
                [start[0], end[0]], [start[1], end[1]],
                color='red', linewidth=2, alpha=0.5
            )
        self.fig_long_2d.canvas.draw()

    def update_long_slice(self, value0=None):
        self.update_long_2d()
        value = self.long_slider.get()
        value = min(self.long_max - 1, value)
        value = max(self.long_min, value)
        self.long_slider.set(value)
        self.update_long_2d()
    
    def update_image_slice(self, value0=None):

        if not hasattr(self, 'image_stack'):
            return
        self.update_long_2d()
        min_value, value, max_value = self.twoSlider.get_values()
        value = min(self.max_crop, value)
        value = max(self.min_crop, value)
        self.twoSlider.set_values(min_value, value, max_value)
        slice_index = int(value)
        self.ax_slice.clear()
        mri_slice = self.image_stack[slice_index]
        try:
            seg_slice = self.seg_display_slice(value, mri_slice.shape)
            inner_mask = seg_slice == 2
            outer_mask = seg_slice == 1
            imat_mask = seg_slice == 3
            normalized_slice = normalize(mri_slice)
            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
            rgb_slice = rgb_slice * 0.8
            rgb_slice[inner_mask, 0] += 0.5
            rgb_slice[outer_mask, 1] += 0.5
            rgb_slice[imat_mask, 2] += 0.5
            rgb_slice = np.clip(rgb_slice, 0, 1)
            rgb_image = (rgb_slice * 255).astype(np.uint8)
            resolution_ = self.dicom_short_info[value][4]
            scale_ = resolution_[0] * resolution_[1]
            print('scale_: ', scale_)
            '''
            plt.subplot(1, 3, 1)
            plt.imshow(inner_mask, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(outer_mask, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(imat_mask, cmap='gray')
            plt.show()
            '''
            inner_area_ = np.sum(inner_mask) * scale_
            outer_area_ = np.sum(outer_mask) * scale_
            fatty_area_ = np.sum(imat_mask) * scale_
            print(self.dicom_short_info[value])
            print('inner_area (mm^2): ', inner_area_)
            print('outer_area (mm^2): ', outer_area_)
            print('fatty_area (mm^2): ', fatty_area_)
        except:
            rgb_image = mri_slice
        self.ax_slice.imshow(rgb_image, cmap='gray')
        self.ax_slice.set_title(f"Slice at Z = {self.z_positions[slice_index]:.2f}")
        self.ax_slice.set_axis_off()
        self.fig_slice.canvas.draw()
        self.update_fat_plot()
        self.update_cache()
    
    def slice_spacing_mm(self):
        """Distance (mm) between neighbouring short-axis slices (see dicom_io.slice_spacing_mm)."""
        return slice_spacing_mm(self.dicom_short_info)

    def update_fat_plot(self):
        value = self.twoSlider.get_values()[1]
        current_slice = int(value)
        if self.dicom_short_info is None or self.segmentation is None:
            return
        info = self.dicom_short_info
        self.fat_plot.clear()
        if self.segmentation.shape[2] != len(info):
            # e.g. a seg file made from another series: its slices don't match the loaded ones
            self.fat_plot.set_title(f"Segmentation has {self.segmentation.shape[2]} slices, images have {len(info)}")
            self.fat_plot.figure.canvas.draw()
            return
        # Per-slice areas (cm^2) of labels 1 (green), 2 (red), 3 (blue)
        sat_mass = np.zeros(self.segmentation.shape[2])
        vat_mass = np.zeros(self.segmentation.shape[2])
        imat_mass = np.zeros(self.segmentation.shape[2])
        slice_index = np.linspace(0, self.segmentation.shape[2] - 1, self.segmentation.shape[2])
        for i in range(self.segmentation.shape[2]):
            inner_mask = self.segmentation[:, :, i] == 2
            outer_mask = self.segmentation[:, :, i] == 1
            imat_mask = self.segmentation[:, :, i] == 3
            pixel_cm2 = float(info[i][4][0]) * float(info[i][4][1]) * 1e-2
            sat_mass[i] = np.sum(outer_mask) * pixel_cm2
            vat_mass[i] = np.sum(inner_mask) * pixel_cm2
            imat_mass[i] = np.sum(imat_mask) * pixel_cm2
        # Volumes (cm^3) over the crop range: each slice's area times the slice spacing
        crop = slice(self.min_crop, self.max_crop + 1)
        depth_cm = self.slice_spacing_mm() * 0.1
        total_sat_mass = np.sum(sat_mass[crop]) * depth_cm
        total_vat_mass = np.sum(vat_mass[crop]) * depth_cm
        total_imat_mass = np.sum(imat_mass[crop]) * depth_cm
        self.fat_plot.plot(slice_index, sat_mass, 'g', label=f"Green: {sat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        self.fat_plot.plot(slice_index, vat_mass, 'r', label=f"Red: {vat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        self.fat_plot.plot(slice_index, imat_mass, 'b', label=f"Blue: {imat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        upValue = max(np.ceil(np.max([np.max(sat_mass), np.max(vat_mass), np.max(imat_mass)]) / 10) * 10, 10)
        self.fat_plot.plot([self.min_crop, self.min_crop], [0, upValue], 'b-.', linewidth=1)
        self.fat_plot.plot([self.max_crop, self.max_crop], [0, upValue], 'r-.', linewidth=1)
        self.fat_plot.scatter(current_slice, sat_mass[current_slice], c='g')
        self.fat_plot.scatter(current_slice, vat_mass[current_slice], c='r')
        self.fat_plot.scatter(current_slice, imat_mass[current_slice], c='k')
        self.fat_plot.set_xlabel("Slice Index")
        self.fat_plot.set_ylabel("Area (cm$^2$)")
        self.fat_plot.legend()
        self.fat_plot.set_title("Volume, slices {}-{}: Green {:.1f}, Red {:.1f}, Blue {:.1f} (cm$^3$)".format(
            self.min_crop, self.max_crop, total_sat_mass, total_vat_mass, total_imat_mass))
        self.fat_plot.grid(True)
        self.fat_plot.set_xlim(0, self.segmentation.shape[2] - 1)
        self.fat_plot.set_xlim(self.min_crop, self.max_crop)
        self.fat_plot.set_ylim(0, upValue)
        self.fat_plot.set_aspect('auto')
        self.fat_plot.figure.canvas.draw()
    
    def update_cache(self):
        pass  # Placeholder for cache update if needed
    
    def update_two_values(self):
        if not hasattr(self, 'twoSlider'):
            return
        min_val, value, max_val = self.twoSlider.get_values()
        if min_val != self.min_crop or max_val != self.max_crop:
            try:
                self.update_fat_plot()
                self.update_long_2d()
            except:
                pass
        self.min_crop = min_val
        self.max_crop = max_val
        if self.twoSlider.get_values()[1] > self.max_crop:
            self.twoSlider.set_values(min_val, self.max_crop, max_val)
        if self.twoSlider.get_values()[1] < self.min_crop:
            self.twoSlider.set_values(min_val, self.min_crop, max_val)
        self.twoSlider.update_labels()
        self.after_id = self.root.after(100, self.update_two_values)
    
    def show_image_popup(self):
        if not hasattr(self, 'image_stack'):
            messagebox.showwarning("Warning", "No image stack loaded.")
            return
        popup = tk.Toplevel(self.root)
        self.image_popup = popup   # closed when the mode is switched
        popup.title("Image Viewer with Drawing")
        popup_fig = plt.figure(figsize=(4, 4))
        popup_ax = popup_fig.add_subplot(111)
        slice_index = self.twoSlider.get_values()[1]
        mri_slice = self.image_stack[slice_index]
        try:
            seg_slice = self.seg_display_slice(slice_index, mri_slice.shape)
            inner_mask = seg_slice == 2
            outer_mask = seg_slice == 1
            imat_mask = seg_slice == 3
            normalized_slice = normalize(mri_slice)
            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
            rgb_slice = rgb_slice * 0.8
            rgb_slice[inner_mask, 0] += 0.5
            rgb_slice[outer_mask, 1] += 0.5
            rgb_slice[imat_mask, 2] += 0.5
            rgb_slice = np.clip(rgb_slice, 0, 1)
            rgb_image = (rgb_slice * 255).astype(np.uint8)
        except:
            rgb_image = mri_slice
        popup_ax.imshow(rgb_image, cmap='gray')
        popup_ax.set_title(f"Slice at Z = {self.z_positions[slice_index]:.2f}")
        popup_ax.set_axis_off()
        popup_canvas = FigureCanvasTkAgg(popup_fig, master=popup)
        popup_canvas.draw()
        popup_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        nav_frame = tk.Frame(popup)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        load_prev_button = tk.Button(nav_frame, text='prev', command=lambda: self.load_previous_slice(), font=font.Font(size=14))
        load_prev_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        load_next_button = tk.Button(nav_frame, text='next', command=lambda: self.load_next_slice(), font=font.Font(size=14))
        load_next_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.popup_ax = popup_ax
        self.popup_canvas = popup_canvas
        self.polygon_points = []  # Store polygon vertices
        self.current_polygon_line = None  # Current polygon being drawn
        self.polygon_color = "white"  # Always use white for polygon drawing
        self.current_mode = "red"  # Default to red mode
        self.drawn_patches = []  # Store drawn patches for potential undo
        self.drawing_polygon = False
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.brush_size = 10
        self.brush_color = "red"
        self.segmentation_mode = "polygon"  # Default to polygon mode
        popup_canvas.mpl_connect("button_press_event", self.on_mouse_press)
        popup_canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        popup_canvas.mpl_connect("button_release_event", self.on_mouse_release)
        popup_canvas.mpl_connect("scroll_event", self.adjust_brush_size)
        button_frame = tk.Frame(popup, bg="lightgray", bd=2, relief=tk.RAISED)
        button_frame.place(relx=0.05, rely=0.05, anchor=tk.NW)

        try:
            polygon_img = load_icon("polygon.png", 30)
            brush_img = load_icon("brush.png", 30)
        except Exception:
            polygon_img = None
            brush_img = None

        mode_frame = tk.Frame(button_frame, bg="lightgray")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.polygon_mode_button = tk.Button(mode_frame, image=polygon_img if polygon_img else None, text="Polygon" if not polygon_img else "",
                                             command=lambda: self.set_segmentation_mode("polygon"), relief=tk.SUNKEN, bd=3)
        self.polygon_mode_button.image = polygon_img
        self.polygon_mode_button.pack(side=tk.LEFT, padx=2)

        self.brush_mode_button = tk.Button(mode_frame, image=brush_img if brush_img else None, text="Brush" if not brush_img else "",
                                           command=lambda: self.set_segmentation_mode("brush"), relief=tk.RAISED, bd=2)
        self.brush_mode_button.image = brush_img
        self.brush_mode_button.pack(side=tk.LEFT, padx=2)
        clear_button = tk.Button(button_frame, text="Clear", bg="black", fg="white", command=lambda: self.set_polygon_mode("clear"))
        clear_button.pack(fill=tk.X, padx=5, pady=2)
        green_button = tk.Button(button_frame, text="Green", bg="green", fg="white", command=lambda: self.set_polygon_mode("green"))
        green_button.pack(fill=tk.X, padx=5, pady=2)
        red_button = tk.Button(button_frame, text="Red", bg="red", fg="white", command=lambda: self.set_polygon_mode("red"))
        red_button.pack(fill=tk.X, padx=5, pady=2)
        blue_button = tk.Button(button_frame, text="Blue", bg="blue", fg="white", command=lambda: self.set_polygon_mode("blue"))
        blue_button.pack(fill=tk.X, padx=5, pady=2)
        self.overlay_var = tk.IntVar(value=2)  # Default to "Over Green"
        over_clear = tk.Radiobutton(button_frame, text="Over Clear", variable=self.overlay_var, value=0, command=lambda: self.set_over_image(0))
        over_clear.pack(anchor=tk.W)
        over_red = tk.Radiobutton(button_frame, text="Over Red", variable=self.overlay_var, value=1, command=lambda: self.set_over_image(2))
        over_red.pack(anchor=tk.W)
        over_green = tk.Radiobutton(button_frame, text="Over Green", variable=self.overlay_var, value=2, command=lambda: self.set_over_image(1))
        over_green.pack(anchor=tk.W)
        over_blue = tk.Radiobutton(button_frame, text="Over Blue", variable=self.overlay_var, value=3, command=lambda: self.set_over_image(3))
        over_blue.pack(anchor=tk.W)

        # Set default overlay mode to "Over Green"
        self.set_over_image(1)
        update_seg = tk.Button(button_frame, text="Update Seg", bg="orange", fg="white", command=self.update_short_slice)
        update_seg.pack(fill=tk.X, pady=5)
        self.undo_button = tk.Button(button_frame, text="Undo (Ctrl+Z)", command=self.undo_edit)
        self.undo_button.pack(fill=tk.X, pady=2)
        self.update_undo_button()
        popup.bind('<Control-z>', lambda e: self.undo_edit())
        popup.bind('<Control-Z>', lambda e: self.undo_edit())
        close_button = tk.Button(button_frame, text="Close", command=popup.destroy)
        close_button.pack(fill=tk.X, pady=5)
    

    def load_previous_slice(self):
        totalNum = self.image_stack.shape[0]
        value = self.twoSlider.get_values()[1]
        self.twoSlider.set_values(self.twoSlider.get_values()[0], (value - 1 + totalNum) % totalNum, self.twoSlider.get_values()[2])
        slice_index = int(value)
        
        mri_slice = self.image_stack[slice_index]
        self.update_popup_image(slice_index)
        self.update_fat_plot()
        self.update_short_slice()
        return
    
    def load_next_slice(self):
        totalNum = self.image_stack.shape[0]
        value = self.twoSlider.get_values()[1]
        self.twoSlider.set_values(self.twoSlider.get_values()[0], (value + 1 + totalNum) % totalNum, self.twoSlider.get_values()[2])
        slice_index = int(value)
        
        mri_slice = self.image_stack[slice_index]
        self.update_popup_image(slice_index)
        self.update_fat_plot()
        self.update_short_slice()
        return
    
    def update_short_slice(self):
        slice_index = self.twoSlider.get_values()[1]
        self.update_popup_image(slice_index)
        self.update_image_slice()
    
    def update_popup_image(self, slice_index):
        if not hasattr(self, 'popup_ax') or not self.popup_ax:
            return
        self.popup_ax.clear()
        mri_slice = self.image_stack[slice_index]
        try:
            seg_slice = self.seg_display_slice(slice_index, mri_slice.shape)
            inner_mask = seg_slice == 2
            outer_mask = seg_slice == 1
            imat_mask = seg_slice == 3
            normalized_slice = normalize(mri_slice)
            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
            rgb_slice = rgb_slice * 0.8
            rgb_slice[inner_mask, 0] += 0.5
            rgb_slice[outer_mask, 1] += 0.5
            rgb_slice[imat_mask, 2] += 0.5
            rgb_slice = np.clip(rgb_slice, 0, 1)
            rgb_image = (rgb_slice * 255).astype(np.uint8)
        except:
            rgb_image = mri_slice
        self.popup_ax.imshow(rgb_image, cmap='gray')
        for patch in self.drawn_patches:
            self.popup_ax.add_patch(patch)
        self.popup_ax.set_title(f"Slice at Z = {self.z_positions[slice_index]:.2f}")
        self.popup_ax.set_axis_off()
        self.popup_canvas.draw()
    
    def set_segmentation_mode(self, mode):
        self.segmentation_mode = mode
        if mode == "polygon":
            self.polygon_mode_button.config(relief=tk.SUNKEN, bd=3)
            self.brush_mode_button.config(relief=tk.RAISED, bd=2)
            self.drawing = False
            self.last_x = None
            self.last_y = None
            if getattr(self, 'cursor_ring', None) is not None and self.cursor_ring in self.popup_ax.patches:
                self.cursor_ring.remove()
            self.cursor_ring = None
            self.activate_new_polygon()   # start drawing with the current colour, no extra click
        else:
            self.polygon_mode_button.config(relief=tk.RAISED, bd=2)
            self.brush_mode_button.config(relief=tk.SUNKEN, bd=3)
            self.reset_polygon()
            self.drawing_polygon = False
            if self.current_mode == "green":
                self.brush_color = "green"
            elif self.current_mode == "red":
                self.brush_color = "red"
            elif self.current_mode == "blue":
                self.brush_color = "blue"
            else:
                self.brush_color = "black"

    def on_mouse_press(self, event):
        if self.segmentation_mode == "polygon":
            self.on_polygon_click(event)
        else:
            self.on_brush_press(event)

    def on_mouse_motion(self, event):
        if self.segmentation_mode == "polygon":
            self.on_polygon_motion(event)
        else:
            self.on_brush_motion(event)

    def on_mouse_release(self, event):
        if self.segmentation_mode == "brush":
            self.on_brush_release(event)

    def on_brush_press(self, event):
        if event.inaxes == self.popup_ax and event.xdata is not None and event.ydata is not None:
            self.begin_edit(self.twoSlider.get_values()[1])   # one brush stroke = one undo step
            self.drawing = True
            self.last_x = event.xdata
            self.last_y = event.ydata
            self.draw(event)

    def on_brush_motion(self, event):
        if self.drawing:
            self.draw(event)
        else:
            self.update_cursor_ring(event)

    def on_brush_release(self, event):
        if not self.drawing:
            return   # release without a stroke (e.g. after a click outside the image)
        self.end_edit()
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.update_popup_image(self.twoSlider.get_values()[1])
        self.update_image_slice()
        self.update_fat_plot()

    def on_polygon_click(self, event):
        """Handle mouse clicks for polygon drawing."""
        if event.inaxes != self.popup_ax or event.xdata is None or event.ydata is None:
            return

        # Only handle clicks if polygon drawing is active
        if not self.drawing_polygon:
            return

        if event.dblclick:
            # Double click closes the polygon
            self.close_polygon()
        else:
            # Single click adds a point
            self.add_polygon_point(event.xdata, event.ydata)

    def on_polygon_motion(self, event):
        """Handle mouse motion for polygon preview."""
        if not self.drawing_polygon or not self.polygon_points or event.inaxes != self.popup_ax:
            return

        if event.xdata is not None and event.ydata is not None:
            self.update_polygon_preview(event.xdata, event.ydata)

    def draw(self, event):
        # print('draw')
        """Draw on the canvas while the mouse is moving."""
        if self.segmentation is None:
            return
        if self.drawing and event.xdata is not None and event.ydata is not None:
            # Keep the brush inside the image shown in the popup
            rSize, cSize = self.image_stack[self.twoSlider.get_values()[1]].shape
            x = min(max(event.xdata, 0), cSize - 1)
            y = min(max(event.ydata, 0), rSize - 1)

            # Store the current axis limits
            xlim = self.popup_ax.get_xlim()
            ylim = self.popup_ax.get_ylim()
            # print('xy lim: ', xlim, ylim)
            # print('last: ', self.last_x, self.last_y)
            if self.last_x is not None and self.last_y is not None:
                # Draw a line from the last position to the current position, as wide as the
                # brush (linewidth is in points, the brush size in image pixels)
                ax_px = self.popup_ax.get_window_extent().width / abs(xlim[1] - xlim[0])
                width_pt = self.brush_size * ax_px * 72 / self.popup_ax.figure.dpi
                self.popup_ax.plot(
                    [self.last_x, x], [self.last_y, y],
                    color=self.brush_color, linewidth=width_pt, solid_capstyle="round"
                )
                # Draw a circle at the current position
                circle = plt.Circle(
                    (x, y), self.brush_size / 2, color=self.brush_color, fill=True
                )
                self.popup_ax.add_patch(circle)

                # Update the segmentation
                self.update_segmentation(self.last_x, self.last_y, x, y)

                # Restore the axis limits
                self.popup_ax.set_xlim(xlim)
                self.popup_ax.set_ylim(ylim)

                # Redraw when Tk is idle: mouse moves come faster than full redraws
                self.popup_canvas.draw_idle()

            self.last_x = x
            self.last_y = y

            # self.update_image_slice()
    def adjust_brush_size(self, event):
        """Adjust the brush size using the mouse wheel."""
        if event.button == "up":
            self.brush_size += 1
        elif event.button == "down":
            self.brush_size -= 1
        # Ensure the brush size doesn't go below 1
        self.brush_size = max(1, self.brush_size)
        

    def update_cursor_ring(self, event):
        """Update the ring around the cursor to indicate the brush size."""
        if event.xdata is not None and event.ydata is not None:
            if not hasattr(self, 'cursor_ring') or self.cursor_ring not in self.popup_ax.patches:
                # Create a new cursor ring if it doesn't exist or isn't in patches
                self.cursor_ring = plt.Circle(
                    (event.xdata, event.ydata), self.brush_size / 2,
                    color=self.brush_color, fill=False, linestyle="--", linewidth=2
                )
                self.popup_ax.add_patch(self.cursor_ring)
            else:
                # Update the existing cursor ring's position and size
                self.cursor_ring.center = (event.xdata, event.ydata)
                self.cursor_ring.set_radius(self.brush_size / 2)
            self.popup_canvas.draw_idle()
        

    def set_over_image(self, num):
        # print(f"Over image: {num}")
        self.over_image = num
        
    def set_polygon_mode(self, mode):
        """Set the polygon drawing mode and activate new polygon drawing."""
        self.current_mode = mode
        self.polygon_color = "white"

        if mode == "green":
            self.brush_color = "green"
        elif mode == "red":
            self.brush_color = "red"
        elif mode == "blue":
            self.brush_color = "blue"
        else:
            self.brush_color = "black"

        print(f"Polygon mode set to: {mode}, color: {self.polygon_color}")

        self.reset_polygon()
        self.activate_new_polygon()

    def activate_new_polygon(self):
        """Activate a new polygon drawing session."""
        # Clear any existing polygon (its line may already be gone if the slice was redrawn)
        self.reset_polygon()

        # Set drawing state to active
        self.drawing_polygon = True

        # Refresh canvas to show clean state
        self.popup_canvas.draw()

        print(f"New polygon activated for mode: {self.current_mode}")

    def add_polygon_point(self, x, y):
        """Add a point to the current polygon."""
        self.polygon_points.append((x, y))
        self.update_polygon_display()

    def update_polygon_preview(self, x, y):
        """Update the polygon preview line."""
        if self.current_polygon_line:
            try:
                self.current_polygon_line.remove()
            except:
                # If removal fails, clear the line from the axis
                if self.current_polygon_line in self.popup_ax.lines:
                    self.popup_ax.lines.remove(self.current_polygon_line)
            self.current_polygon_line = None

        if len(self.polygon_points) > 0:
            # Draw lines from all existing points to current mouse position
            xs = [p[0] for p in self.polygon_points] + [x]
            ys = [p[1] for p in self.polygon_points] + [y]
            self.current_polygon_line, = self.popup_ax.plot(xs, ys,
                                                           color=self.polygon_color,
                                                           linestyle='--', alpha=0.7)
            self.popup_canvas.draw_idle()

    def update_polygon_display(self):
        """Update the display to show current polygon points."""
        if self.current_polygon_line:
            try:
                self.current_polygon_line.remove()
            except:
                # If removal fails, clear the line from the axis
                if self.current_polygon_line in self.popup_ax.lines:
                    self.popup_ax.lines.remove(self.current_polygon_line)
            self.current_polygon_line = None

        if len(self.polygon_points) > 1:
            xs = [p[0] for p in self.polygon_points]
            ys = [p[1] for p in self.polygon_points]
            self.current_polygon_line, = self.popup_ax.plot(xs, ys,
                                                           color=self.polygon_color,
                                                           marker='o', markersize=4,
                                                           linewidth=2)
        elif len(self.polygon_points) == 1:
            x, y = self.polygon_points[0]
            self.current_polygon_line, = self.popup_ax.plot([x], [y],
                                                           color=self.polygon_color,
                                                           marker='o', markersize=4)
        self.popup_canvas.draw_idle()

    def close_polygon(self):
        """Close the current polygon and apply it to segmentation."""
        if len(self.polygon_points) < 3:
            return  # Need at least 3 points for a polygon

        # Close the polygon visually
        if self.current_polygon_line:
            try:
                self.current_polygon_line.remove()
            except:
                # If removal fails, clear the line from the axis
                if self.current_polygon_line in self.popup_ax.lines:
                    self.popup_ax.lines.remove(self.current_polygon_line)

        # Add first point to close the polygon
        closed_points = self.polygon_points + [self.polygon_points[0]]
        xs = [p[0] for p in closed_points]
        ys = [p[1] for p in closed_points]

        self.current_polygon_line, = self.popup_ax.plot(xs, ys,
                                                       color=self.polygon_color,
                                                       linewidth=2)

        # Apply polygon to segmentation
        self.apply_polygon_to_segmentation()

        # Automatically update segmentation display
        self.update_popup_image(self.twoSlider.get_values()[1])
        self.update_image_slice()
        self.update_fat_plot()

        # Reset for next polygon and deactivate drawing
        self.reset_polygon()
        self.drawing_polygon = False
        self.popup_canvas.draw()

        print("Polygon completed and applied to segmentation")

    def reset_polygon(self):
        """Reset the current polygon."""
        self.polygon_points = []
        if self.current_polygon_line:
            try:
                self.current_polygon_line.remove()
            except:
                # If removal fails, clear the line from the axis
                if self.current_polygon_line in self.popup_ax.lines:
                    self.popup_ax.lines.remove(self.current_polygon_line)
            self.current_polygon_line = None

    def apply_polygon_to_segmentation(self):
        """Apply the drawn polygon to the segmentation data."""
        if getattr(self, 'segmentation', None) is None or len(self.polygon_points) < 3:
            return
        from matplotlib.path import Path

        slice_index = self.twoSlider.get_values()[1]
        rows, cols = self.image_stack[slice_index].shape   # the popup shows the image as is
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        inside_mask = Path(self.polygon_points).contains_points(coords).reshape(rows, cols)
        self.begin_edit(slice_index)   # one polygon = one undo step
        self.paint_segmentation(inside_mask, slice_index)
        self.end_edit()

    def paint_segmentation(self, display_mask, slice_index):
        """Relabel the pixels under display_mask (popup image coordinates).

        Pixels that carry the "Over ..." label (0 clear, 1 green, 2 red, 3 blue) get the label
        of the current mode; all other pixels are left alone.
        """
        new_label = {"clear": 0, "green": 1, "red": 2, "blue": 3}.get(self.current_mode)
        if new_label is None:
            return
        seg_slice = self.segmentation[:, :, slice_index]   # view: edits write through
        # Inverse of the overlay orientation used in update_popup_image
        mask = self.to_seg_orientation(display_mask)
        if mask.shape != seg_slice.shape:
            return   # segmentation doesn't belong to this image
        seg_slice[mask & (seg_slice == self.over_image)] = new_label

    def update_segmentation(self, x0, y0, x1, y1):
        """Paint the brush stroke from (x0, y0) to (x1, y1) (popup image coordinates)."""
        if getattr(self, 'segmentation', None) is None:
            return
        slice_index = self.twoSlider.get_values()[1]
        rows, cols = self.image_stack[slice_index].shape
        # Only the box around the stroke can be painted: compute the distances there
        r = self.brush_size / 2
        r0, r1 = max(int(np.floor(min(y0, y1) - r)), 0), min(int(np.ceil(max(y0, y1) + r)) + 1, rows)
        c0, c1 = max(int(np.floor(min(x0, x1) - r)), 0), min(int(np.ceil(max(x0, x1) + r)) + 1, cols)
        if r0 >= r1 or c0 >= c1:
            return
        yy, xx = np.mgrid[r0:r1, c0:c1]
        # Distance from each pixel to the stroke segment; within the brush radius is painted
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        dist2 = (xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2
        mask = np.zeros((rows, cols), bool)
        mask[r0:r1, c0:c1] = dist2 <= r * r
        self.paint_segmentation(mask, slice_index)

    # --- Undo of manual edits (brush strokes and polygons in the Show Image window)

    UNDO_STEPS = 30   # edits kept for undo

    def begin_edit(self, slice_index):
        """Keep a copy of the slice before an edit, for undo."""
        if getattr(self, 'segmentation', None) is None or slice_index >= self.segmentation.shape[2]:
            return
        stack = getattr(self, 'undo_stack', [])
        # Steps of a segmentation that has since been replaced (new Seg run, file loaded, flip) no longer apply
        stack = [e for e in stack if e[0] is self.segmentation][-(self.UNDO_STEPS - 1):]
        stack.append((self.segmentation, slice_index, self.segmentation[:, :, slice_index].copy()))
        self.undo_stack = stack

    def end_edit(self):
        """Forget the last snapshot if the edit changed nothing (e.g. no pixel had the "Over" label)."""
        stack = getattr(self, 'undo_stack', [])
        if stack:
            seg, k, before = stack[-1]
            if seg is self.segmentation and np.array_equal(seg[:, :, k], before):
                stack.pop()
        self.update_undo_button()

    def undo_edit(self):
        """Undo the last brush stroke or polygon; shows the slice it was made on."""
        stack = [e for e in getattr(self, 'undo_stack', []) if e[0] is self.segmentation]
        if not stack or self.drawing:
            self.undo_stack = stack
            self.update_undo_button()
            return
        _, k, before = stack.pop()
        self.undo_stack = stack
        self.segmentation[:, :, k] = before
        lo, _, hi = self.twoSlider.get_values()
        self.twoSlider.set_values(lo, k, hi)   # show the restored slice (edits are made inside the crop range)
        self.update_short_slice()
        self.update_undo_button()

    def update_undo_button(self):
        button = getattr(self, 'undo_button', None)
        if button is not None and button.winfo_exists():
            n = sum(1 for e in getattr(self, 'undo_stack', []) if e[0] is self.segmentation)
            button.config(state=tk.NORMAL if n else tk.DISABLED, text=f"Undo (Ctrl+Z){f' [{n}]' if n else ''}")

WINDOW_ASPECT = 1.6   # main window width : height at start (it stays resizable)


def initial_geometry(root, aspect=WINDOW_ASPECT):
    """Window size at start: 88% of the screen height, `aspect` times as wide (at most 95% of the screen width)."""
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    h = int(sh * 0.88)
    w = min(int(h * aspect), int(sw * 0.95))
    return f"{w}x{h}+{max((sw - w) // 2, 0)}+0"


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(initial_geometry(root))
    app = DicomViewerApp(root)
    root.mainloop()
