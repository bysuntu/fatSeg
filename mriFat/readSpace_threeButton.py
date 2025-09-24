import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, font
import nibabel as nib
from PIL import Image, ImageTk
#from seg import segment
from sat_seg import auto_segment

def segment(slice_data):

    return None, None, None, None

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

def readDicomFile(file_path, seriesName):
    try:
        ds = dicom.dcmread(file_path)

        # Check if this file has pixel data
        if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
            return None

        pixel_array = ds.pixel_array
        patient = getattr(ds, 'PatientName', 'Unknown')
        series_name = getattr(ds, 'SeriesDescription', '')

        # If a specific series is requested, check if this file matches
        if seriesName:
            if seriesName.lower() not in series_name.lower():
                return None

        # Check for required DICOM attributes
        if not hasattr(ds, 'ImageOrientationPatient') or not hasattr(ds, 'ImagePositionPatient'):
            print(f"Missing required DICOM attributes in {file_path}")
            return None

        image_orientation = ds.ImageOrientationPatient
        image_position = ds.ImagePositionPatient
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
        slice_position = float(getattr(ds, 'SliceLocation', 0.0))

        return [pixel_array, patient, series_name, image_orientation, image_position, pixel_spacing, slice_thickness, slice_position]
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {str(e)}")
        return None

def parseDicomFolder(folder_path, seriesName=None):
    dicom_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            # Skip DICOMDIR and other non-image files
            if f.upper() in ['DICOMDIR', 'DICOMDIR.', 'VERSION'] or f.startswith('.'):
                continue
            dicom_files.append(os.path.join(root, f))

    print(f"Found {len(dicom_files)} potential DICOM files")
    if not dicom_files:
        raise ValueError("No files found in the selected folder.")

    all_pixel, all_info = [], []
    valid_series = set()

    for file in dicom_files:
        result = readDicomFile(file, seriesName)
        if result is not None:
            pixel_, patient, series_name, *info = result
            all_pixel.append(pixel_)
            all_info.append([patient, series_name] + info)
            valid_series.add(series_name)

    print(f"Found {len(all_pixel)} valid DICOM files with pixel data")
    if valid_series:
        print(f"Available series: {list(valid_series)}")

    if not all_pixel:
        if seriesName:
            raise ValueError(f"No valid DICOM files found for series '{seriesName}'. Available series: {list(valid_series) if valid_series else 'None'}")
        else:
            raise ValueError("No valid DICOM files with pixel data found in the folder.")
    
    sorted_pixel_info = sorted([[all_pixel[k], all_info[k][6]] for k in range(len(all_info))], key=lambda x: float(x[1]))
    all_pixel = [row[0] for row in sorted_pixel_info]
    all_info = sorted(all_info, key=lambda x: float(x[6]))
    return all_pixel, all_info

class DicomViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dicom Viewer GUI")
        self.root.configure(background='black')
        self.click_state = 1  # Start with SAT mode (state 1)
        self.threshold = 0
        self.segmentation_threshold = 100
        self.cache = None
        self.dicom_long_info = None
        self.dicom_long_pixels = None
        self.dicom_short_info = None
        self.dicom_short_pixels = None
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
            self.sat_img = ImageTk.PhotoImage(Image.open("sat.png").resize((180, 180), Image.Resampling.LANCZOS))
            self.thigh_img = ImageTk.PhotoImage(Image.open("thigh.png").resize((180, 180), Image.Resampling.LANCZOS))
        except Exception as e:
            # print(f"Error loading images: {str(e)}")
            self.sat_img = self.thigh_img = None
        
        self.switch_button = tk.Button(left_frame, image=self.sat_img, text="click to choose", command=self.update_button_image)
        self.switch_button.pack(pady=5)
        
        # Sub-frames
        self.left_top_frame = tk.Frame(left_frame, bg="#FFF0F0", width=180, height=160)
        self.left_top_frame.pack(padx=5, pady=5)
        self.left_top_frame.pack_propagate(False)
        
        self.left_mid_frame = tk.Frame(left_frame, bg="#F0F0FF", width=180, height=80)
        self.left_mid_frame.pack(padx=5, pady=5)
        self.left_mid_frame.pack_propagate(False)
        
        self.left_bottom_frame = tk.Frame(left_frame, bg="#F0FFF0", width=180, height=80)
        self.left_bottom_frame.pack(padx=5, pady=5)
        self.left_bottom_frame.pack_propagate(False)
        
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
        
        # Long axis view
        self.fig_long_2d = plt.figure(figsize=(4, 4))
        self.ax_long_2d = self.fig_long_2d.add_subplot(111)
        canvas_long_2d = FigureCanvasTkAgg(self.fig_long_2d, top_frame)
        canvas_long_2d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax_long_2d.spines['top'].set_visible(False)
        self.ax_long_2d.spines['right'].set_visible(False)
        self.ax_long_2d.spines['bottom'].set_visible(False)
        self.ax_long_2d.spines['left'].set_visible(False)
        self.ax_long_2d.xaxis.set_ticks([])
        self.ax_long_2d.yaxis.set_ticks([])
        
        self.long_slider = tk.Scale(top_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Slider")
        self.long_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        self.long_slider.bind("<ButtonRelease-1>", self.update_long_slice)
        
        # Slice view
        self.fig_slice = plt.figure(figsize=(4, 4))
        self.ax_slice = self.fig_slice.add_subplot(111)
        canvas_slice = FigureCanvasTkAgg(self.fig_slice, bottom_frame)
        canvas_slice.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Fat plot
        self.fig_plot = plt.figure(figsize=(4, 4))
        self.fat_plot = self.fig_plot.add_subplot(111)
        canvas_plot = FigureCanvasTkAgg(self.fig_plot, bottom_frame)
        canvas_plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Three button slider
        self.twoSlider = ThreeButtonSlider(bottom_frame, 0, 10, 0, 5, 10, 200, 20, parent=self)
        self.twoSlider.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10, padx=(10, 10))
        
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
        self.switch_button.config(state=tk.DISABLED)  # Lock the button

        # Start periodic update
        self.update_two_values()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        # Cancel the after callback
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        plt.close('all')
        self.root.quit()
        self.root.destroy()
    
    def update_button_image(self):
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
        elif self.click_state == 1:
            self.switch_button.config(image=self.thigh_img)
            self.click_state = 2
            self.draw_line_button.pack(pady=5)
            self.thigh_button.pack(pady=5)
            self.auto_seg_button.pack_forget()
        else:
            self.switch_button.config(image=self.sat_img)
            self.click_state = 1
            self.draw_line_button.pack_forget()
            self.thigh_button.pack_forget()
            self.auto_seg_button.pack(pady=5)

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

    def load_short_files(self):
        folder = filedialog.askdirectory(title="Select DICOM Directory")
        if not folder:
            return
        try:
            # self.dicom_short_pixels, self.dicom_short_info = parseDicomFolder(folder, '6pt_DIXON_VIBE_F')
            self.dicom_short_pixels, self.dicom_short_info = parseDicomFolder(folder) #, '6pt_DIXON_VIBE_F')
            self.image_stack = np.array(self.dicom_short_pixels)
            # print('image stack shape: ', self.image_stack.shape)
            self.min_crop = 0
            self.max_crop = self.image_stack.shape[0] - 1
            self.current = int(self.max_crop / 2)
            self.twoSlider.set_values2(self.min_crop, self.max_crop)
            self.twoSlider.set_values(self.min_crop, self.current, self.max_crop)
            self.z_positions = [info[6] for info in self.dicom_short_info]
            self.threshold_low = np.percentile(self.image_stack, 1)
            self.threshold_high = np.percentile(self.image_stack, 99)
            self.image_stack = np.clip(self.image_stack, self.threshold_low, self.threshold_high)
            self.update_image_slice()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load short axis slices: {str(e)}")
    
    def load_long_files(self):
        folder = filedialog.askdirectory(title="Select DICOM Directory")
        if not folder:
            return
        try:
            self.dicom_long_pixels, self.dicom_long_info = parseDicomFolder(folder) #, 'localizer_cor')
            self.dicom_long_pixels = np.array(self.dicom_long_pixels)
            self.long_min = 0
            self.long_max = self.dicom_long_pixels.shape[0]
            self.long_slider.config(from_=0, to=self.long_max - 1)
            self.update_long_slice()
        except Exception as e:
            messagebox.showerror("Error", f"Will display long axis slices after loading short axis slices") #: {str(e)}")
    
    def load_segmentation_files(self):
        filename = filedialog.askopenfilename(title="Select Segmentation File", filetypes=[("NII.GZ files", "*.nii.gz")])
        if not filename:
            return
        try:
            self.affine = nib.load(filename).affine
            self.segmentation = nib.load(filename).get_fdata()
            self.update_image_slice()
            self.update_fat_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load segmentation file: {str(e)}")
    
    def save_segmentation(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".nii.gz",
            filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")],
            title="Save Segmentation As"
        )
        if file_path:
            if not file_path.endswith(".nii.gz"):
                file_path += ".nii.gz"
            try:
                seg = nib.Nifti1Image(self.segmentation, self.affine if hasattr(self, 'affine') else np.eye(4))
                nib.save(seg, file_path)
                messagebox.showinfo("Success", f"Segmentation saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save segmentation: {str(e)}")
    
    def auto_segment(self):
        if not hasattr(self, 'image_stack'):
            messagebox.showwarning("Warning", "No image stack loaded.")
            return
        self.segmentation = auto_segment(self.image_stack, self.segmentation_threshold)
        self.update_image_slice()
        self.update_fat_plot()
    
    def thigh_mode(self):
        if self.click_state != 2:
            messagebox.showwarning("Mode Incorrect", "Thigh segmentation is only available in Thigh Seg mode.")
            return

        sliceId = self.twoSlider.get_values()[1]

        if self.dicom_short_info is None or self.dicom_short_pixels is None:
            # print("No short axis slices loaded.")
            return
        target_slice = self.dicom_short_pixels[sliceId]

        fatSeg = []    
        for i in range(len(self.dicom_short_pixels)):
            cur_ = self.dicom_short_pixels[i]
            seg1, seg2, muscle, *_ = segment(cur_)
            fatSeg.append(seg2 * 2 + seg1 + muscle * 3)

        self.segmentation = np.array(fatSeg).transpose(2, 1, 0)

        self.update_image_slice()
        self.update_short_slice()
        self.update_popup_image(sliceId)
        return
    
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
            middle_point_3d = current_long_info[3] + middle_point[0] * np.array(current_long_info[2][:3]) * current_long_info[4][0] + middle_point[1] * np.array(current_long_info[2][3:]) * current_long_info[4][1]
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
            inner_mask = self.segmentation[:, :, value] == 2
            outer_mask = self.segmentation[:, :, value] == 1
            imat_mask = self.segmentation[:, :, value] == 3
            if inner_mask.shape != mri_slice.shape:
                inner_mask = inner_mask.T
                outer_mask = outer_mask.T
                imat_mask = imat_mask.T
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
    
    def update_fat_plot(self):
        value = self.twoSlider.get_values()[1]
        current_slice = int(value)
        if self.dicom_short_info is None or self.segmentation is None:
            return
        info = self.dicom_short_info
        self.fat_plot.clear()
        sat_mass = np.zeros(self.segmentation.shape[2])
        vat_mass = np.zeros(self.segmentation.shape[2])
        imat_mass = np.zeros(self.segmentation.shape[2])
        total_sat_vol = np.sum(self.segmentation[:, :, self.min_crop:self.max_crop + 1] == 1)
        total_sat_mass = total_sat_vol * info[0][4][0] * info[0][4][1] * 1e-2
        total_vat_vol = np.sum(self.segmentation[:, :, self.min_crop:self.max_crop + 1] == 2)
        total_vat_mass = total_vat_vol * info[0][4][0] * info[0][4][1] * 1e-2
        total_imat_vol = np.sum(self.segmentation[:, :, self.min_crop:self.max_crop + 1] == 3)
        total_imat_mass = total_imat_vol * info[0][4][0] * info[0][4][1] * 1e-2
        slice_index = np.linspace(0, self.segmentation.shape[2] - 1, self.segmentation.shape[2])
        for i in range(self.segmentation.shape[2]):
            inner_mask = self.segmentation[:, :, i] == 2
            outer_mask = self.segmentation[:, :, i] == 1
            imat_mask = self.segmentation[:, :, i] == 3
            sat_mass[i] = np.sum(outer_mask) * info[i][4][0] * info[i][4][1] * 1e-2
            vat_mass[i] = np.sum(inner_mask) * info[i][4][0] * info[i][4][1] * 1e-2
            imat_mass[i] = np.sum(imat_mask) * info[i][4][0] * info[i][4][1] * 1e-2
        self.fat_plot.plot(slice_index, sat_mass, 'g', label=f"Green: {sat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        self.fat_plot.plot(slice_index, vat_mass, 'r', label=f"Red: {vat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        self.fat_plot.plot(slice_index, imat_mass, 'b', label=f"Blue: {imat_mass[current_slice]:.2f} cm$^2$", linewidth=0.5)
        upValue = np.ceil(np.max([np.max(sat_mass), np.max(vat_mass)]) / 10) * 10
        self.fat_plot.plot([self.min_crop, self.min_crop], [0, upValue], 'b-.', linewidth=1)
        self.fat_plot.plot([self.max_crop, self.max_crop], [0, upValue], 'r-.', linewidth=1)
        self.fat_plot.scatter(current_slice, sat_mass[current_slice], c='g')
        self.fat_plot.scatter(current_slice, vat_mass[current_slice], c='r')
        self.fat_plot.scatter(current_slice, imat_mass[current_slice], c='k')
        self.fat_plot.set_xlabel("Slice Index")
        self.fat_plot.set_ylabel("Mass (g)")
        self.fat_plot.legend()
        self.fat_plot.set_title("Total Green: {:.2f} , Red: {:.2f}, Blue: {:.2f} (cm$^2$)".format(total_sat_mass, total_vat_mass, total_imat_mass))
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
            messagebox.showwarningcerr("Warning", "No image stack loaded.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Image Viewer with Drawing")
        popup_fig = plt.figure(figsize=(4, 4))
        popup_ax = popup_fig.add_subplot(111)
        slice_index = self.twoSlider.get_values()[1]
        mri_slice = self.image_stack[slice_index]
        try:
            inner_mask = self.segmentation[:, :, slice_index] == 2
            outer_mask = self.segmentation[:, :, slice_index] == 1
            imat_mask = self.segmentation[:, :, slice_index] == 3
            if inner_mask.shape != mri_slice.shape:
                inner_mask = inner_mask.T
                outer_mask = outer_mask.T
                imat_mask = imat_mask.T
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
        popup_canvas.mpl_connect("button_press_event", self.on_polygon_click)
        popup_canvas.mpl_connect("motion_notify_event", self.on_polygon_motion)
        button_frame = tk.Frame(popup, bg="lightgray", bd=2, relief=tk.RAISED)
        button_frame.place(relx=0.05, rely=0.05, anchor=tk.NW)
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
            inner_mask = self.segmentation[:, :, slice_index] == 2
            outer_mask = self.segmentation[:, :, slice_index] == 1
            imat_mask = self.segmentation[:, :, slice_index] == 3
            if inner_mask.shape != mri_slice.shape:
                inner_mask = inner_mask.T
                outer_mask = outer_mask.T
                imat_mask = imat_mask.T
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
        try:
            [rSize, cSize, zSize] = self.segmentation.shape
        except AttributeError:
            return
        # print('segmentation: ', self.segmentation.shape)
        if self.drawing and event.xdata is not None and event.ydata is not None:
            # print('drawing on image')
            x, y = event.xdata, event.ydata

            # Adjust bounds checking based on flip state
            if self.flip_axes:
                # When flipped, image dimensions are swapped for display
                x = max(min(x, cSize - self.brush_size // 2 - 1), self.brush_size // 2 + 1)
                y = max(min(y, rSize - self.brush_size // 2 - 1), self.brush_size // 2 + 1)
            else:
                x = max(min(x, rSize - self.brush_size // 2 - 1), self.brush_size // 2 + 1)
                y = max(min(y, cSize - self.brush_size // 2 - 1), self.brush_size // 2 + 1)
            '''
            # Store the current axis limits
            xlim = self.ax_slice.get_xlim()
            ylim = self.ax_slice.get_ylim()

            if self.last_x and self.last_y:
                # Draw a line from the last position to the current position
                self.ax_slice.plot(
                    [self.last_x, x], [self.last_y, y],
                    color=self.brush_color, linewidth=self.brush_size, solid_capstyle="round"
                )
                # Draw a circle at the current position
                circle = plt.Circle(
                    (x, y), self.brush_size / 2, color=self.brush_color, fill=True
                )
                self.ax_slice.add_patch(circle)

                # Update the segmentation
                self.update_segmentation(x, y)

                # Restore the axis limits
                self.ax_slice.set_xlim(xlim)
                self.ax_slice.set_ylim(ylim)

                # Redraw the canvas
                self.popup_canvas.draw()

            self.last_x = x
            self.last_y = y

            # self.update_image_slice()
            '''
            # Store the current axis limits
            xlim = self.popup_ax.get_xlim()
            ylim = self.popup_ax.get_ylim()

            if self.last_x and self.last_y:
                # Draw a line from the last position to the current position
                self.popup_ax.plot(
                    [self.last_x, x], [self.last_y, y],
                    color=self.brush_color, linewidth=self.brush_size, solid_capstyle="round"
                )
                # Draw a circle at the current position
                circle = plt.Circle(
                    (x, y), self.brush_size / 2, color=self.brush_color, fill=True
                )
                self.popup_ax.add_patch(circle)

                # Update the segmentation
                self.update_segmentation(x, y)

                # Restore the axis limits
                self.popup_ax.set_xlim(xlim)
                self.popup_ax.set_ylim(ylim)

                # Redraw the canvas
                # self.popup_ax.draw()
                self.popup_canvas.draw()

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
            self.popup_canvas.draw()
        

    def set_over_image(self, num):
        # print(f"Over image: {num}")
        self.over_image = num
        
    def set_polygon_mode(self, mode):
        """Set the polygon drawing mode and activate new polygon drawing."""
        self.current_mode = mode
        # Always use white color for polygon drawing
        self.polygon_color = "white"

        print(f"Polygon mode set to: {mode}, color: {self.polygon_color}")  # Debug print

        # Reset current polygon when changing modes
        self.reset_polygon()

        # Activate new polygon drawing session
        self.activate_new_polygon()

    def activate_new_polygon(self):
        """Activate a new polygon drawing session."""
        # Clear any existing polygon
        self.polygon_points = []
        if self.current_polygon_line:
            self.current_polygon_line.remove()
            self.current_polygon_line = None

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
        if not hasattr(self, 'segmentation') or len(self.polygon_points) < 3:
            return

        from matplotlib.path import Path
        import numpy as np

        # Get current slice index
        slice_index = self.twoSlider.get_values()[1]

        # Create a path from polygon points
        path = Path(self.polygon_points)

        # Get segmentation shape
        seg_shape = self.segmentation.shape

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:seg_shape[1], 0:seg_shape[0]]
        coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

        # Adjust coordinates if axes are flipped
        if self.flip_axes:
            coords = np.column_stack((coords[:, 1], coords[:, 0]))

        # Find points inside the polygon
        inside_mask = path.contains_points(coords)
        inside_mask = inside_mask.reshape(seg_shape[1], seg_shape[0])

        # Apply segmentation based on mode and overlay settings
        self.apply_polygon_segmentation(inside_mask, slice_index)

    def apply_polygon_segmentation(self, mask, slice_index):
        """Apply polygon mask to segmentation based on current mode and overlay settings."""
        if self.flip_axes:
            mask = mask.T

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:  # If point is inside polygon
                    if self.over_image == 0:  # on clear
                        if self.current_mode == "green":
                            self.segmentation[i, j, slice_index] = 1 if self.segmentation[i, j, slice_index] == 0 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "red":
                            self.segmentation[i, j, slice_index] = 2 if self.segmentation[i, j, slice_index] == 0 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "blue":
                            self.segmentation[i, j, slice_index] = 3 if self.segmentation[i, j, slice_index] == 0 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "clear":
                            self.segmentation[i, j, slice_index] = 0 if self.segmentation[i, j, slice_index] == 0 else self.segmentation[i, j, slice_index]

                    elif self.over_image == 1:  # on Green
                        if self.current_mode == "clear":
                            self.segmentation[i, j, slice_index] = 0 if self.segmentation[i, j, slice_index] == 1 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "red":
                            self.segmentation[i, j, slice_index] = 2 if self.segmentation[i, j, slice_index] == 1 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "blue":
                            self.segmentation[i, j, slice_index] = 3 if self.segmentation[i, j, slice_index] == 1 else self.segmentation[i, j, slice_index]

                    elif self.over_image == 2:  # on red
                        if self.current_mode == "clear":
                            self.segmentation[i, j, slice_index] = 0 if self.segmentation[i, j, slice_index] == 2 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "green":
                            self.segmentation[i, j, slice_index] = 1 if self.segmentation[i, j, slice_index] == 2 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "blue":
                            self.segmentation[i, j, slice_index] = 3 if self.segmentation[i, j, slice_index] == 2 else self.segmentation[i, j, slice_index]

                    elif self.over_image == 3:  # on blue
                        if self.current_mode == "clear":
                            self.segmentation[i, j, slice_index] = 0 if self.segmentation[i, j, slice_index] == 3 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "green":
                            self.segmentation[i, j, slice_index] = 1 if self.segmentation[i, j, slice_index] == 3 else self.segmentation[i, j, slice_index]
                        elif self.current_mode == "red":
                            self.segmentation[i, j, slice_index] = 2 if self.segmentation[i, j, slice_index] == 3 else self.segmentation[i, j, slice_index]

    def update_segmentation(self, x, y):
        """Update the segmentation data based on the brush mode."""
        if not hasattr(self, 'segmentation'):
            return  # No segmentation data loaded

        # Get the current slice index
        slice_index = self.twoSlider.get_values()[1]

        # Convert coordinates to pixel indices
        x_index = int(round(x))
        y_index = int(round(y))

        # If segmentation is flipped, swap coordinates for segmentation indexing
        if self.flip_axes:
            seg_x, seg_y = y_index, x_index
        else:
            seg_x, seg_y = x_index, y_index

        # Define the brush radius in pixels
        radius = self.brush_size // 2

        # Update the segmentation data within the brush radius
        for j in range(seg_x - radius, seg_x + radius + 1):
            for i in range(seg_y - radius, seg_y + radius + 1):
                if 0 <= i < self.segmentation.shape[1] and 0 <= j < self.segmentation.shape[0]:
                    # print(f"i: {i}, j: {j}, slice index: {slice_index}")
                    # print(f"self.over_image: {self.over_image}, self.current_mode: {self.current_mode}")
                    if self.over_image == 0: # on clear
                        if self.current_mode == "clear": # on clear
                            # print('draw clear on clear')
                            self.segmentation[j, i, slice_index] = 0 if self.segmentation[j, i, slice_index] == 0 else self.segmentation[j, i, slice_index]
                        elif self.current_mode == "green": # on green
                            print('draw green on clear')
                            self.segmentation[j, i, slice_index] = 1 if self.segmentation[j, i, slice_index] == 0 else self.segmentation[j, i, slice_index]
                        elif self.current_mode == "red": # on red
                            print('draw red on clear')
                            self.segmentation[j, i, slice_index] = 2 if self.segmentation[j, i, slice_index] == 0 else self.segmentation[j, i, slice_index]
                        elif self.current_mode == "blue":
                            # print('draw blue on clear')
                            self.segmentation[j, i, slice_index] = 3 if self.segmentation[j, i, slice_index] == 0 else self.segmentation[j, i, slice_index]
                    
                    elif self.over_image == 1: #on Green 
                        if self.current_mode == "clear":
                            # print('draw clear on green')
                            self.segmentation[j, i, slice_index] = 0  if self.segmentation[j, i, slice_index] == 1 else self.segmentation[j, i, slice_index]  # Clear segmentation
                        # elif self.current_mode == "green":
                            # print('draw green on green')
                            # self.segmentation[j, i, slice_index] = 1  # Green segmentation
                        elif self.current_mode == "red":
                            print('draw red on green')
                            self.segmentation[j, i, slice_index] = 2  if self.segmentation[j, i, slice_index] == 1 else self.segmentation[j, i, slice_index]  # Red segmentation
                        elif self.current_mode == "blue":
                            # print("draw blue on green")
                            self.segmentation[j, i, slice_index] = 3  if self.segmentation[j, i, slice_index] == 1 else self.segmentation[j, i, slice_index] # Blue segmentation
                    
                    
                    elif self.over_image == 2: # on red
                        if self.current_mode == "clear":
                            # print('draw clear on red')
                            self.segmentation[j, i, slice_index] = 0  if self.segmentation[j, i, slice_index] == 2 else self.segmentation[j, i, slice_index]  # Clear segmentation
                        elif self.current_mode == "green":  
                            # print('draw green on red')
                            self.segmentation[j, i, slice_index] = 1  if self.segmentation[j, i, slice_index] == 2 else self.segmentation[j, i, slice_index]  # Green segmentation
                        # elif self.current_mode == "red":
                            # print('draw red on red')
                        elif self.current_mode == "blue":
                            self.segmentation[j, i, slice_index] = 3  if self.segmentation[j, i, slice_index] == 2 else self.segmentation[j, i, slice_index] # Blue segmentation 
        
                    elif self.over_image == 3: # on blue
                        if self.current_mode == "clear":
                            # print('draw clear on blue')
                            self.segmentation[j, i, slice_index] = 0  if self.segmentation[j, i, slice_index] == 3 else self.segmentation[j, i, slice_index]  # Clear segmentation
                        elif self.current_mode == "green":  
                            # print('draw green on blue')
                            self.segmentation[j, i, slice_index] = 1  if self.segmentation[j, i, slice_index] == 3 else self.segmentation[j, i, slice_index]  # Green segmentation
                        # elif self.current_mode == "red":
                            print('draw red on blue')
                        elif self.current_mode == "red":
                            self.segmentation[j, i, slice_index] = 2  if self.segmentation[j, i, slice_index] == 3 else self.segmentation[j, i, slice_index] # Blue segmentation 
        
        
        # print('done: ', self.segmentation.shape)

if __name__ == "__main__":
    root = tk.Tk()
    app = DicomViewerApp(root)
    root.mainloop()
