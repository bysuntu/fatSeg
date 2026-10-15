"""
DICOM series loading, shared by the GUI (readSpace_threeButton.py), pipeline.py and the scripts.

parseDicomFolder() returns one acquisition's slices sorted along the slice normal, with per-slice
info [patient, series name, ImageOrientationPatient, ImagePositionPatient, PixelSpacing,
SliceThickness, SliceLocation, SeriesInstanceUID, SeriesNumber, SeriesTime].
dicom_affine() and slice_spacing_mm() give the geometry of such a stack.
"""
import os

import numpy as np
import pydicom as dicom


def _name_matches(series_name, seriesName=None, seriesSuffix=None, seriesContains=None):
    """Whether a SeriesDescription passes the filter: exact name, suffix, substring (no case); all set ones must hold."""
    if seriesName and seriesName.lower() != series_name.lower():
        return False
    if seriesSuffix and not series_name.strip().lower().endswith(seriesSuffix.lower()):
        return False
    if seriesContains and seriesContains.lower() not in series_name.lower():
        return False
    return True


def _series_description(file_path):
    """SeriesDescription of a file, reading only that tag; None if it is not a DICOM file."""
    try:
        return str(getattr(dicom.dcmread(file_path, stop_before_pixels=True, specific_tags=['SeriesDescription']),
                           'SeriesDescription', ''))
    except Exception:
        return None


def readDicomFile(file_path, seriesName, seriesSuffix=None, seriesContains=None):
    try:
        ds = dicom.dcmread(file_path)
        series_name = str(getattr(ds, 'SeriesDescription', ''))

        # If a specific series is requested, check if this file matches (before decoding pixels)
        if not _name_matches(series_name, seriesName, seriesSuffix, seriesContains):
            return None

        # Check if this file has pixel data
        if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
            return None

        pixel_array = ds.pixel_array
        patient = getattr(ds, 'PatientName', 'Unknown')

        # Check for required DICOM attributes
        if not hasattr(ds, 'ImageOrientationPatient') or not hasattr(ds, 'ImagePositionPatient'):
            print(f"Missing required DICOM attributes in {file_path}")
            return None

        image_orientation = ds.ImageOrientationPatient
        image_position = ds.ImagePositionPatient
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
        slice_position = float(getattr(ds, 'SliceLocation', 0.0))
        # Which acquisition the file belongs to (several can share a SeriesDescription)
        series_uid = str(getattr(ds, 'SeriesInstanceUID', ''))
        series_number = getattr(ds, 'SeriesNumber', None)
        series_time = str(getattr(ds, 'SeriesTime', ''))

        return [pixel_array, patient, series_name, image_orientation, image_position, pixel_spacing, slice_thickness, slice_position,
                series_uid, series_number, series_time]
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {str(e)}")
        return None

class MultipleSeriesError(Exception):
    """More than one acquisition matches and no series was chosen (not a ValueError: callers
    use ValueError for 'no matching series' and would try another sequence instead)."""


class SeriesChoiceCancelled(Exception):
    """The user closed the series choice without picking one."""


def _slice_order_key(info):
    """Position of a slice along its own normal (ImagePositionPatient . (row x column direction))."""
    orient = np.array(info[2], float)
    return float(np.dot(np.array(info[3], float), np.cross(orient[:3], orient[3:])))


def parseDicomFolder(folder_path, seriesName=None, seriesSuffix=None, seriesContains=None, choose_series=None):
    """Pixels and info of the slices in folder_path matching the series filter, sorted along the slice normal.

    info per slice: [patient, series name, ImageOrientationPatient, ImagePositionPatient, PixelSpacing,
    SliceThickness, SliceLocation, SeriesInstanceUID, SeriesNumber, SeriesTime].
    Only one acquisition is returned. When several match (e.g. a repeated scan with the same name),
    choose_series(options) picks one: options is a list of dicts (number, name, time, slices, shape) and
    it returns an index into it, or None to cancel (SeriesChoiceCancelled). Without choose_series this
    raises MultipleSeriesError.
    """
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

    # With a series filter, read only the series name of each file first and the whole file only
    # for the wanted series (a series folder often holds several series: in, opp, water, fat, ...)
    found = None
    if seriesName or seriesSuffix or seriesContains:
        names = {file: _series_description(file) for file in dicom_files}
        found = {n for n in names.values() if n is not None}
        dicom_files = [file for file, n in names.items()
                       if n is not None and _name_matches(n, seriesName, seriesSuffix, seriesContains)]

    for file in dicom_files:
        result = readDicomFile(file, seriesName, seriesSuffix, seriesContains)
        if result is not None:
            pixel_, patient, series_name, *info = result
            all_pixel.append(pixel_)
            all_info.append([patient, series_name] + info)
            valid_series.add(series_name)

    print(f"Found {len(all_pixel)} valid DICOM files with pixel data")
    if valid_series:
        print(f"Available series: {list(valid_series)}")

    if not all_pixel:
        if seriesName or seriesSuffix or seriesContains:
            # List what the folder does contain, to make a naming mismatch obvious
            wanted = seriesName or (f"*{seriesContains}*" if seriesContains else f"*{seriesSuffix}")
            raise ValueError(f"No valid DICOM files found for series '{wanted}'. Series in folder: {sorted(found) if found else 'None'}")
        else:
            raise ValueError("No valid DICOM files with pixel data found in the folder.")
    
    # One acquisition only: slices of different scans must not be interleaved in one stack
    uids = list(dict.fromkeys(info[7] for info in all_info))
    if len(uids) > 1:
        options = []
        for uid in uids:
            members = [k for k in range(len(all_info)) if all_info[k][7] == uid]
            first = all_info[members[0]]
            options.append(dict(uid=uid, number=first[8], name=first[1], time=first[9], slices=len(members),
                                shape=tuple(np.shape(all_pixel[members[0]]))))
        options.sort(key=lambda o: (o['number'] is None, o['number'] if o['number'] is not None else 0, o['time']))
        listing = '; '.join(f"#{o['number']} '{o['name']}' {o['slices']} slices {o['shape'][0]}x{o['shape'][1]}"
                            for o in options)
        if choose_series is None:
            raise MultipleSeriesError(f"{len(options)} series match in {folder_path}: {listing}")
        index = choose_series(options)
        if index is None:
            raise SeriesChoiceCancelled()
        keep = options[index]['uid']
        print(f"Using series #{options[index]['number']} of: {listing}")
        members = [k for k in range(len(all_info)) if all_info[k][7] == keep]
        all_pixel = [all_pixel[k] for k in members]
        all_info = [all_info[k] for k in members]

    # Sort by position along the slice normal (SliceLocation can be missing or 0)
    order = sorted(range(len(all_info)), key=lambda k: _slice_order_key(all_info[k]))
    return [all_pixel[k] for k in order], [all_info[k] for k in order]


def dicom_affine(info):
    """NIfTI affine (RAS) of an array laid out like the slices: (rows, cols, slices).

    Voxel (i, j, k) is ImagePositionPatient of slice 0 + i rows along the column direction
    (IOP[3:], PixelSpacing[0]) + j columns along the row direction (IOP[:3], PixelSpacing[1])
    + k slice steps (first to last slice position); DICOM LPS is turned into NIfTI RAS.
    """
    orient = np.array(info[0][2], float)
    row_dir, col_dir = orient[:3], orient[3:]
    first = np.array(info[0][3], float)
    if len(info) > 1:
        step = (np.array(info[-1][3], float) - first) / (len(info) - 1)
    else:
        step = np.cross(row_dir, col_dir) * float(info[0][5])
    lps = np.eye(4)
    lps[:3, 0] = col_dir * float(info[0][4][0])
    lps[:3, 1] = row_dir * float(info[0][4][1])
    lps[:3, 2] = step
    lps[:3, 3] = first
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def slice_spacing_mm(info):
    """Distance (mm) between neighbouring slices, along the slice normal.

    The median gap between slice positions (ImagePositionPatient); SliceThickness when there
    is only one slice or the positions don't give a usable gap.
    """
    thickness = float(info[0][5])
    if len(info) < 2:
        return thickness
    orient = np.array(info[0][2], float)
    normal = np.cross(orient[:3], orient[3:])
    pos = np.array([np.dot(np.array(k[3], float), normal) for k in info])
    gaps = np.abs(np.diff(pos))
    gaps = gaps[gaps > 0.1 * thickness]   # repeated positions (duplicate slices) are not gaps
    return float(np.median(gaps)) if gaps.size else thickness
