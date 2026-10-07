# Installing fatSeg

These steps set up fatSeg on Windows, where it's developed and tested. They take about 15–30 minutes, most of it downloading packages.

## Before you start

| | Requirement |
|---|---|
| Operating system | Windows 10 or 11, 64-bit |
| Python | 3.12, 64-bit (tested with 3.12.10). 3.10 and 3.11 should also work |
| Disk space | about 3 GB (TensorFlow alone is over 1 GB) |
| Memory | 8 GB recommended. The large 512×512 abdominal series and the U-Net both need a few GB |
| GPU | not needed. Everything runs on the CPU |
| Internet | needed once, to download the packages |
| Model weights | `unet_thighfat_finetuned.keras`, obtained separately (only for thigh **AI Seg**) |

## Step 1: Install Python 3.12 (64-bit)

Choose **one** of the following.

### Option A: python.org installer

1. Download the **Windows installer (64-bit)** for Python 3.12 from <https://www.python.org/downloads/windows/>.
2. Run it and tick **"Add python.exe to PATH"** on the first screen.
3. Choose **Customize installation** and make sure **"tcl/tk and IDLE"** is ticked. The fatSeg window is built with Tkinter, which is part of tcl/tk.
4. Finish the installation.

### Option B: WinPython (portable, no administrator rights)

1. Download a **64-bit WinPython 3.12** from <https://winpython.github.io/>.
2. Unpack it to a short path, e.g. `E:\WPy64-312100`. The original fatSeg setup was made this way.
3. Python is then at `E:\WPy64-312100\python\python.exe`. Use that full path wherever these instructions say `python`.

### Check

Open **Command Prompt** (`cmd`) and run:

```bat
python --version
python -c "import struct, tkinter; print(struct.calcsize('P') * 8, 'bit, Tk', tkinter.TkVersion)"
```

You should see `Python 3.12.x` and `64 bit, Tk 8.6`. If you see `32 bit` or a Tkinter error, reinstall Python as described above.

## Step 2: Get the code

**With git:**

```bat
cd /d E:\
git clone https://github.com/bysuntu/fatSeg.git
cd fatSeg
```

**Without git:** on the GitHub page click **Code → Download ZIP**, unpack it, e.g. to `E:\fatSeg`, and open a Command Prompt in that folder:

```bat
cd /d E:\fatSeg
```

Keep the path short and free of special characters. Long paths can break the TensorFlow installation (see [Troubleshooting](#troubleshooting)).

All following commands are run **from the project folder** (the one containing `runFat.bat`).

## Step 3: Create the virtual environment

```bat
python -m venv .seg
```

With WinPython:

```bat
E:\WPy64-312100\python\python.exe -m venv .seg
```

The environment **must be called `.seg` and be inside the project folder**: `runFat.bat` starts the app with `.seg\Scripts\python.exe`.

## Step 4: Install the packages

```bat
.seg\Scripts\python.exe -m pip install --upgrade pip
.seg\Scripts\python.exe -m pip install -r requirements.txt
```

This downloads about 1.5 GB and can take several minutes. Wait until the prompt returns. A final line starting with `Successfully installed` means it worked.

`requirements.txt` installs:

| Package | Used for |
|---|---|
| pydicom | reading DICOM images |
| pylibjpeg, pylibjpeg-libjpeg | reading DICOMs stored with JPEG (e.g. JPEG Lossless) compression |
| numpy, scipy, scikit-image, opencv-python | image processing and segmentation |
| shapely, pygeoops | geometry helpers used by the thigh method |
| nibabel | reading and writing segmentations (`.nii.gz`) |
| matplotlib, Pillow | display in the GUI |
| tensorflow | the thigh U-Net (**AI Seg**) and `finetune.py` |

The commands call `.seg\Scripts\python.exe` directly, so you never need to "activate" the environment. That avoids PowerShell's script-execution restrictions.

## Step 5: Add the model weights (thigh AI Seg only)

The trained U-Net is **not stored in the repository** because each file is about 93 MB. Obtain it from the project maintainer and copy it into the project folder, next to `runFat.bat`:

```
fatSeg\unet_thighfat_finetuned.keras
```

- **Which file the GUI loads:** `unet_thighfat_finetuned.keras`, falling back to `unet_thighfat_segmentation_model_best_loss.keras` if it's missing.
- **Without either file:** abdomen mode and the classical **Thigh Seg** still work. Only **AI Seg** (and **Combined**, which builds on it) won't.

## Step 6: Check the installation

```bat
.seg\Scripts\python.exe -c "import tensorflow, pydicom, nibabel, cv2, skimage, scipy, shapely, pygeoops, PIL, matplotlib, tkinter; print('OK')"
```

The last line should read `OK`. TensorFlow may print informational messages before it (e.g. about oneDNN), which can be ignored.

## Step 7: Start fatSeg

Double-click **`runFat.bat`** in the project folder, or run it from the Command Prompt:

```bat
runFat.bat
```

The first start can take 10–20 seconds while TensorFlow loads. The window opens in abdomen mode (belly icon at the top left). See [README.md](README.md) for how to use it.

Always start it from the project folder. The mode icons (`sat.png`, `thigh.png`) are loaded from the current folder.

## Folder layout after installation

```
fatSeg\
├── .seg\                              virtual environment (step 3, not in git)
├── mriFat\                            application code
│   ├── readSpace_threeButton.py       the GUI
│   ├── abd_seg.py                     abdominal SAT/VAT segmentation
│   └── seg.py                         classical thigh segmentation
├── unet_thighfat_finetuned.keras      model weights (step 5, not in git)
├── runFat.bat                         starts the app
├── requirements.txt
├── compare_overlap.py, finetune.py, …
└── sat.png, thigh.png, brush.png, polygon.png   GUI icons
```

## Updating

```bat
cd /d E:\fatSeg
git pull
.seg\Scripts\python.exe -m pip install -r requirements.txt
```

Re-running the install command only adds or updates packages that changed.

## Starting over

To rebuild the environment from scratch, delete the `.seg` folder and repeat steps 3, 4 and 6:

```bat
rmdir /s /q .seg
```

This only removes the installed packages, not your code, model weights or data.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `'python' is not recognized` | Python isn't on PATH. Reinstall with "Add python.exe to PATH" ticked, or use the full path to `python.exe` (WinPython). |
| `No matching distribution found for tensorflow` | Python is 32-bit, too old or too new for TensorFlow. Use 64-bit Python 3.10–3.12. |
| TensorFlow install fails with an `OSError` about a path or file name | Windows' 260-character path limit. Move the project to a short path (e.g. `E:\fatSeg`) and recreate `.seg`, or enable long paths: set `LongPathsEnabled` to `1` under `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` (administrator rights needed) and restart. |
| `ModuleNotFoundError: No module named 'tkinter'` | Python was installed without tcl/tk. Rerun the installer → Modify → tick "tcl/tk and IDLE", or use WinPython. Then recreate `.seg`. |
| `runFat.bat`: "The system cannot find the path specified" | `.seg` is missing, misnamed or not in the project folder. Repeat step 3, or start `runFat.bat` from the project folder. |
| `ModuleNotFoundError` for any other package | Step 4 didn't finish. Run it again and check for errors. |
| AI Seg: "Model file not found" | Copy the `.keras` file into the project folder (step 5). |
| Mode icons show text ("click to choose") instead of pictures | The app was started from another folder. Use `runFat.bat` from the project folder. |
| Downloads fail behind a hospital or company proxy | Add `--proxy http://user:password@proxy:port` to the `pip install` commands, or ask IT for the proxy address. |

## Linux and macOS (untested)

The code itself is plain Python, but it has only been used on Windows, and `runFat.bat` is Windows-only. On Linux or macOS the equivalent steps would be:

```bash
python3 -m venv .seg
.seg/bin/python -m pip install -r requirements.txt
.seg/bin/python mriFat/readSpace_threeButton.py
```

On Linux, Tkinter usually comes as a separate system package (e.g. `sudo apt install python3-tk`).
