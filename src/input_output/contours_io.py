import os
import json
import glob

import numpy as np
from loguru import logger

from version import version_file_str
from gui.popup_windows.message_boxes import ErrorMessage
from input_output.read_xml import read_xml
from input_output.write_xml import write_xml


def read_contours(main_window, file_name=None):
    """Reads contours saved in json/xml format and displays the contours in the graphics scene"""
    success = False
    json_files = glob.glob(f'{file_name}_contours*.json')
    xml_files = glob.glob(f'{file_name}_contours*.xml')

    if not main_window.config.save.use_xml_files and json_files:  # json files have priority over xml unless desired
        newest_json = max(json_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_json}')
        with open(newest_json, 'r') as in_file:
            main_window.data = json.load(in_file)
        if 'measures' not in main_window.data:  # added in version 0.4.5
            main_window.data['measures'] = [[None, None] for _ in range(main_window.metadata['num_frames'])]
        if 'reference' not in main_window.data:  # added in version 0.7.3
            main_window.data['reference'] = [None] * main_window.metadata['num_frames']
        if 'gating_signal' not in main_window.data:  # added in version 0.7.4
            main_window.data['gating_signal'] = {}
        if 'angles' not in main_window.data: # added in version 1.1.2
            main_window.data['angles'] = [[None, None] for _ in range(main_window.metadata['num_frames'])]
        success = True

    elif xml_files:
        newest_xml = max(xml_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_xml}')
        read_xml(main_window, newest_xml)
        main_window.data['lumen'] = map_to_list(main_window.data['lumen'])
        for key in [
            'lumen_area',
            'lumen_circumf',
            'longest_distance',
            'shortest_distance',
            'elliptic_ratio',
            'vector_length',
            'vector_angle',
        ]:
            main_window.data[key] = [0] * main_window.metadata[
                'num_frames'
            ]  # initialise empty containers for data not stored in xml
        for key in ['lumen_centroid', 'farthest_point', 'nearest_point']:
            main_window.data[key] = (
                [[] for _ in range(main_window.metadata['num_frames'])],
                [[] for _ in range(main_window.metadata['num_frames'])],
            )  # initialise empty containers for data not stored in xml
        success = True

    if success:
        main_window.contours_drawn = True
        contour_type = getattr(main_window, "ContourType", "lumen")
        if contour_type in main_window.data:
            main_window.display.set_data(main_window.data[contour_type], main_window.images)
        else:
            main_window.display.set_data(main_window.data['lumen'], main_window.images)
        main_window.hide_contours_box.setChecked(False)

    return success


def _to_serializable(obj):
    """Simple helper passed to json.dump to handle numpy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # fallback
    try:
        return str(obj)
    except Exception:
        return None


def write_contours(main_window):
    """Writes contours to a json/xml file.

    - If main_window.config.save.use_xml_files is True: write legacy XML for lumen
      (keeps compatibility) AND write a JSON sidecar that contains all contour layers.
    - Otherwise: write a JSON containing all of main_window.data (serialized).
    """
    if not main_window.image_displayed:
        ErrorMessage(main_window, "Cannot write contours before reading input file")
        return

    try:
        base = os.path.splitext(main_window.file_name)[0]
    except Exception:
        base = getattr(main_window, "file_name", "contours_output")
        base = os.path.splitext(base)[0]

    version_str = globals().get("version_file_str", version_file_str)
    json_out_path = f"{base}_contours_{version_str}.json"

    # Ensure main_window.data exists and keys are in sensible form
    data = getattr(main_window, "data", {}) or {}

    if main_window.config.save.use_xml_files:
        # Legacy XML export for lumen (keeps previous behaviour)
        # Build x,y lists frame-by-frame from lumen data (defensive)
        num_frames = main_window.metadata.get("num_frames", 0)
        x = []
        y = []
        lumen_data = data.get("lumen", [[], []])
        lx = lumen_data[0] if len(lumen_data) > 0 else []
        ly = lumen_data[1] if len(lumen_data) > 1 else []

        for frame in range(num_frames):
            if frame < len(lx):
                new_x_lumen = lx[frame] or []
            else:
                new_x_lumen = []
            if frame < len(ly):
                new_y_lumen = ly[frame] or []
            else:
                new_y_lumen = []

            x.append(new_x_lumen)
            y.append(new_y_lumen)

        try:
            write_xml(
                x,
                y,
                main_window.images.shape,
                main_window.metadata.get("resolution"),
                getattr(main_window, "ivusPullbackRate", None),
                data.get("phases"),
                main_window.file_name,
            )
            logger.info(f"Wrote legacy XML for lumen to {main_window.file_name} (and sidecar JSON)")
        except Exception as e:
            logger.exception(f"Failed to write XML contours: {e}")

        try:
            with open(json_out_path, "w") as out_file:
                json.dump(data, out_file, default=_to_serializable, indent=2)
            logger.info(f"Wrote contours JSON sidecar to: {json_out_path}")
        except Exception as e:
            logger.exception(f"Failed to write contours JSON sidecar: {e}")

    else:
        # Write the whole main_window.data to JSON (better safe than sorry)
        try:
            with open(json_out_path, "w") as out_file:
                json.dump(data, out_file, default=_to_serializable, indent=2)
            logger.info(f"Wrote contours JSON to: {json_out_path}")
        except Exception as e:
            logger.exception(f"Failed to write contours JSON: {e}")


def map_to_list(contours):
    """Converts map to list"""
    x, y = contours
    x = [list(x[i]) for i in range(len(x))]
    y = [list(y[i]) for i in range(len(y))]

    return (x, y)


def save_gated_images(main_window, file_name=None):
    """Saves diastolic and systolic images as a 3D numpy array"""
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save gated images before reading the input file.')
        return

    diastolic_images = []
    systolic_images = []

    for frame in range(main_window.metadata['num_frames']):
        if main_window.data['phases'][frame] == 'D':
            diastolic_images.append(main_window.images[frame])
        elif main_window.data['phases'][frame] == 'S':
            systolic_images.append(main_window.images[frame])

    diastolic_images = np.array(diastolic_images)
    systolic_images = np.array(systolic_images)

    out_path_diastolic = os.path.splitext(main_window.file_name)[0] + '_diastolic.npy'
    out_path_systolic = os.path.splitext(main_window.file_name)[0] + '_systolic.npy'
    np.save(out_path_diastolic, diastolic_images)
    np.save(out_path_systolic, systolic_images)
