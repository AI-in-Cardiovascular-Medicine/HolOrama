from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem, QWidget
from domain.io_types import MetaDataCCTA, MetaDataIntravascular

# callable(title, message, default) → float — injected so callers stay testable
PromptFn = Callable[[str, str, float], float]


# ─── Assembly ────────────────────────────────────────────────────────────────


def parse_metadata_dcm(
    df: pd.DataFrame,
    num_frames: int,
    prompt_fn: Optional[PromptFn] = None,
) -> MetaDataIntravascular:
    modality = extract_modality(df)
    patient_name, birthdate, sex = extract_patient_info(df)
    manufacturer, model = extract_manufacturer(df)
    dimension = extract_dimension(df) or num_frames
    pullback_start_frame = extract_pullback_start_frame(df)

    pullback_rate = extract_pullback_rate(df)
    if pullback_rate is None and prompt_fn:
        pullback_rate = prompt_fn('Pullback Speed', 'No pullback speed found, please enter pullback speed (mm/s)', 0.5)

    resolution = extract_resolution(df)
    if resolution is None and prompt_fn:
        resolution = prompt_fn('Pixel Spacing', 'No pixel spacing info found, please enter pixel spacing (mm)', 0.01)

    pullback_length: float | np.ndarray | None
    frame_rate: float | None
    if modality == 'OCT':
        frame_time_ms = extract_frame_time_ms(df)
        if frame_time_ms is not None:
            frame_rate = round(1000 / frame_time_ms, 2)
        elif prompt_fn:
            frame_rate = prompt_fn('Frame Rate', 'No frame rate found, enter frame rate (fps):', 180.0)
        else:
            frame_rate = None
        if pullback_rate and frame_rate:
            duration_s = num_frames / frame_rate
            pullback_length = pullback_rate * duration_s
            # Abbott OPTIS stores FrameTime as ~100ms regardless of actual rate.
            # Derive correct fps from pullback speed per FCC spec (FCC-ID sb6c408650):
            #   36/18 mm/s → 180 fps (standard); 10/20/25 mm/s → 100 fps (C7 Dragonfly)
            if 'abbott' in manufacturer.lower() and 'optis' in model.lower():
                frame_rate = 180.0 if abs(pullback_rate - 36.0) < 1 or abs(pullback_rate - 18.0) < 1 else 100.0
        else:
            pullback_length = None
    else:
        pullback_length = extract_pullback_length_ivus(df, pullback_rate or 0.0, num_frames)
        frame_rate = extract_frame_rate(df)

    return MetaDataIntravascular(
        modality=modality,
        patient_name=patient_name,
        birthdate=birthdate,
        sex=sex,
        pullback_speed=pullback_rate,
        pullback_length=pullback_length,
        resolution=resolution,
        dimension=dimension,
        manufacturer=manufacturer,
        model=model,
        pullback_start_frame=pullback_start_frame,
        frame_rate=frame_rate,
    )


def parse_metadata_nifti(
    df: pd.DataFrame,
    num_frames: int,
    is_oct: bool = False,
    prompt_fn: Optional[PromptFn] = None,
) -> MetaDataIntravascular:
    modality = 'OCT' if is_oct else 'IVUS'
    xy_spacing, z_spacing = extract_nifti_spacing(df)
    dimension = extract_nifti_dimension(df)

    resolution = xy_spacing
    if resolution is None and prompt_fn:
        resolution = prompt_fn(
            'Pixel Spacing', 'No pixel spacing found in NIfTI header, enter pixel spacing (mm):', 0.01
        )

    default_speed = 36.0 if is_oct else 0.5
    pullback_rate = prompt_fn('Pullback Speed', 'Enter pullback speed (mm/s):', default_speed) if prompt_fn else None

    pullback_length: float | np.ndarray | None
    frame_rate: float | None
    if is_oct:
        if z_spacing:
            pullback_length = z_spacing * num_frames
            frame_rate = round(pullback_rate / z_spacing, 2) if pullback_rate else None
        else:
            frame_rate = extract_nifti_frame_rate(df)
            if frame_rate is None and prompt_fn:
                frame_rate = prompt_fn('Frame Rate', 'No frame rate found, enter frame rate (fps):', 180.0)
            duration_s = num_frames / frame_rate if frame_rate else 0.0
            pullback_length = pullback_rate * duration_s if pullback_rate else None
    else:
        pullback_length = np.arange(1, num_frames + 1) * z_spacing if z_spacing else np.zeros(num_frames)
        frame_rate = extract_nifti_frame_rate(df)

    return MetaDataIntravascular(
        modality=modality,
        pullback_speed=pullback_rate,
        pullback_length=pullback_length,
        resolution=resolution,
        dimension=dimension,
        pullback_start_frame=0,
        frame_rate=frame_rate,
    )


# ─── UI layer ────────────────────────────────────────────────────────────────


class MetadataWindow(QMainWindow):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.table = main_window.metadata_table
        self.setWindowTitle('Metadata')
        self._fit_to_table()
        self.setCentralWidget(self.table)

    def _fit_to_table(self) -> None:
        w = sum(self.table.columnWidth(i) for i in range(self.table.columnCount()))
        h = sum(self.table.rowHeight(i) for i in range(self.table.rowCount()))
        self.setFixedSize(w, h)


_CCTA_DISPLAY_FIELDS: list[tuple[str, Callable[[MetaDataCCTA], Optional[str]]]] = [
    ('Modality', lambda m: m.modality),
    ('Patient Name', lambda m: m.patient_name),
    ('Date of Birth', lambda m: m.birthdate),
    ('Sex', lambda m: m.sex),
    ('Manufacturer', lambda m: f'{m.manufacturer} ({m.model})' if m.manufacturer != 'Unknown' else None),
    ('Slice Thickness', lambda m: f'{m.slice_thickness:.3f} mm' if m.slice_thickness else None),
    (
        'Pixel Spacing',
        lambda m: f'{m.pixel_spacing[0]:.3f} × {m.pixel_spacing[1]:.3f} mm' if m.pixel_spacing != (0.0, 0.0) else None,
    ),
]


class CctaMetadataWindow(QMainWindow):
    def __init__(self, parent: QWidget, ccta_metadata: MetaDataCCTA) -> None:
        super().__init__(parent)
        self.setWindowTitle('CCTA Metadata')

        main_rows = [(label, fn(ccta_metadata)) for label, fn in _CCTA_DISPLAY_FIELDS]
        main_rows = [(lbl, v) for lbl, v in main_rows if v is not None]
        extra_rows = list(ccta_metadata.raw_tags.items())

        total = len(main_rows) + 1 + len(extra_rows)  # +1 for '...' separator
        table = QTableWidget(total, 2)

        for i, (label, value) in enumerate(main_rows):
            table.setItem(i, 0, QTableWidgetItem(label))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

        sep = len(main_rows)
        sep_item = QTableWidgetItem('...')
        sep_item.setFlags(Qt.ItemFlag.NoItemFlags)
        table.setItem(sep, 0, sep_item)
        table.setItem(sep, 1, QTableWidgetItem(''))

        for j, (tag_name, tag_value) in enumerate(extra_rows):
            row = sep + 1 + j
            table.setItem(row, 0, QTableWidgetItem(str(tag_name)))
            table.setItem(row, 1, QTableWidgetItem(_fmt_dicom_value(tag_value)))

        h_header = table.horizontalHeader()
        if h_header is not None:
            h_header.hide()
        v_header = table.verticalHeader()
        if v_header is not None:
            v_header.hide()
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        w = sum(table.columnWidth(i) for i in range(table.columnCount()))
        h = min(600, sum(table.rowHeight(i) for i in range(table.rowCount())))
        self.setFixedSize(w, h)
        self.setCentralWidget(table)


# Descriptions consumed by MetaData — excluded from the raw "remaining" section
_METADATA_DESCRIPTIONS = {
    'Modality',
    "Patient's Name",
    'Patient Name',
    "Patient's Birth Date",
    'Patient Birth Date',
    "Patient's Sex",
    'Patient Sex',
    'IVUS Pullback Rate',
    'BostonPullbackRate',
    'Frame Time Vector',
    'Frame Time',
    'Sequence of Ultrasound Regions',
    'Pixel Spacing',
    'Rows',
    'Columns',
    'Manufacturer',
    "Manufacturer's Model Name",
    'Manufacturer Model Name',
    'IVUS Pullback Start Frame Number',
    'Cine Rate',
    'Number of Frames',
    # NIfTI
    'pixdim',
    'dim',
}

_DISPLAY_FIELDS: list[tuple[str, Callable[[MetaDataIntravascular], Optional[str]]]] = [
    ('Modality', lambda m: m.modality),
    ('Patient Name', lambda m: m.patient_name),
    ('Date of Birth', lambda m: m.birthdate),
    ('Sex', lambda m: m.sex),
    ('Pullback Speed', lambda m: f'{m.pullback_speed} mm/s' if m.pullback_speed is not None else None),
    ('Resolution (mm)', lambda m: f'{m.resolution:.4f}' if m.resolution is not None else None),
    ('Dimensions', lambda m: str(m.dimension) if m.dimension is not None else None),
    ('Manufacturer', lambda m: f'{m.manufacturer} ({m.model})' if m.manufacturer else None),
    ('Frame Rate', lambda m: f'{m.frame_rate} fps' if m.frame_rate is not None else 'Unknown'),
    ('Start Frame', lambda m: str(m.pullback_start_frame) if m.pullback_start_frame is not None else None),
]


def populate_metadata_table(
    table: QTableWidget,
    parsed: MetaDataIntravascular,
    full_df: pd.DataFrame,
) -> None:
    main_rows = [(label, fn(parsed)) for label, fn in _DISPLAY_FIELDS]
    main_rows = [(lbl, v) for lbl, v in main_rows if v is not None]

    remaining = full_df[~full_df['Description'].isin(_METADATA_DESCRIPTIONS)][['Description', 'Value']]
    extra_rows = [(d, _fmt_dicom_value(v)) for d, v in zip(remaining['Description'], remaining['Value'])]

    total = len(main_rows) + 1 + len(extra_rows)  # +1 for '...' separator
    table.setRowCount(total)
    table.setColumnCount(2)

    for i, (label, value) in enumerate(main_rows):
        table.setItem(i, 0, QTableWidgetItem(label))
        table.setItem(i, 1, QTableWidgetItem(str(value)))

    sep = len(main_rows)
    sep_item = QTableWidgetItem('...')
    sep_item.setFlags(Qt.ItemFlag.NoItemFlags)
    table.setItem(sep, 0, sep_item)
    table.setItem(sep, 1, QTableWidgetItem(''))

    for j, (label, value) in enumerate(extra_rows):
        row = sep + 1 + j
        table.setItem(row, 0, QTableWidgetItem(str(label)))
        table.setItem(row, 1, QTableWidgetItem(str(value)))

    h_header = table.horizontalHeader()
    if h_header is not None:
        h_header.hide()
    v_header = table.verticalHeader()
    if v_header is not None:
        v_header.hide()
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


# ─── DICOM value formatters ──────────────────────────────────────────────────


def _fmt_dicom_date(val: str) -> str:
    """YYYYMMDD → YYYY/MM/DD, pass through anything else."""
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f'{s[:4]}/{s[4:6]}/{s[6:]}'
    return s


def _fmt_dicom_time(val: str) -> str:
    """HHMMSS or HHMMSS.F (possibly stored as float) → HH:MM:SS."""
    integer_part = str(val).strip().split('.')[0]
    if len(integer_part) == 6 and integer_part.isdigit():
        return f'{integer_part[:2]}:{integer_part[2:4]}:{integer_part[4:]}'
    return str(val)


def _fmt_dicom_value(val) -> str:
    """Best-effort formatting for raw DICOM values in the extra rows."""
    if hasattr(val, '__len__') and not isinstance(val, str):
        items = list(val)
        truncated = items[:5]
        suffix = ', ...' if len(items) > 5 else ''
        return '[' + ', '.join(str(i) for i in truncated) + suffix + ']'
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return _fmt_dicom_date(s)
    integer_part = s.split('.')[0]
    if len(integer_part) == 6 and integer_part.isdigit():
        return _fmt_dicom_time(s)
    return s


# ─── Pure extraction helpers (no PyQt) ───────────────────────────────────────


def _val(df: pd.DataFrame, description: str):
    rows = df[df['Description'] == description]['Value']
    return rows.iloc[0] if not rows.empty else None


_MODALITY_ALIAS_MAP: dict[str, str] = {
    'US': 'IVUS',
    'OPT': 'OCT',
}


def extract_modality(df: pd.DataFrame) -> Optional[str]:
    val = _val(df, 'Modality')
    return _MODALITY_ALIAS_MAP.get(val, val) if val is not None else None


def extract_patient_info(df: pd.DataFrame) -> tuple[str, str, str]:
    # pydicom uses apostrophe in elem.name for some fields
    name = _val(df, "Patient's Name") or _val(df, 'Patient Name') or 'Unknown'
    birth = _fmt_dicom_date(str(_val(df, "Patient's Birth Date") or _val(df, 'Patient Birth Date') or 'Unknown'))
    sex = str(_val(df, "Patient's Sex") or _val(df, 'Patient Sex') or 'Unknown')
    return str(name), birth, sex


def extract_pullback_rate(df: pd.DataFrame) -> Optional[float]:
    for field in ('IVUS Pullback Rate', 'BostonPullbackRate'):
        v = _val(df, field)
        if v is not None:
            return float(v)
    return None


def extract_resolution(df: pd.DataFrame) -> Optional[float]:
    seq = _val(df, 'Sequence of Ultrasound Regions')
    if seq is not None:
        region = seq[0]
        multiplier = 10 if region.PhysicalUnitsXDirection == 3 else 1
        return float(region.PhysicalDeltaX) * multiplier
    spacing = _val(df, 'Pixel Spacing')
    if spacing is not None:
        return float(spacing[0] if hasattr(spacing, '__len__') else spacing)
    return None


def extract_frame_time_vector(df: pd.DataFrame) -> Optional[list[float]]:
    ftv = _val(df, 'Frame Time Vector')
    return [float(f) for f in ftv] if ftv is not None else None


def extract_frame_time_ms(df: pd.DataFrame) -> Optional[float]:
    ft = _val(df, 'Frame Time')
    return float(ft) if ft is not None else None


def extract_pullback_length_ivus(df: pd.DataFrame, pullback_rate: float, num_frames: int) -> np.ndarray:
    ftv = extract_frame_time_vector(df)
    if ftv is not None:
        pullback_time = np.cumsum(ftv) / 1000  # ms → s
        return pullback_time * pullback_rate
    return np.zeros(num_frames)


def extract_frame_rate(df: pd.DataFrame) -> Optional[float]:
    v = _val(df, 'Cine Rate')
    return float(v) if v is not None else None


def extract_dimension(df: pd.DataFrame) -> Optional[int]:
    v = _val(df, 'Rows')
    return int(v) if v is not None else None


def extract_pullback_start_frame(df: pd.DataFrame) -> int:
    v = _val(df, 'IVUS Pullback Start Frame Number')
    return int(v) if v is not None else 0


def extract_manufacturer(df: pd.DataFrame) -> tuple[str, str]:
    mfr = str(_val(df, 'Manufacturer') or 'Unknown')
    model = str(_val(df, "Manufacturer's Model Name") or _val(df, 'Manufacturer Model Name') or 'Unknown')
    return mfr, model


def extract_nifti_spacing(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    pixdim = _val(df, 'pixdim')
    if pixdim is not None and hasattr(pixdim, '__len__') and len(pixdim) > 3:
        xy = float(pixdim[1]) if pixdim[1] > 0 else None
        z = float(pixdim[3]) if pixdim[3] > 0 else None
        return xy, z
    return None, None


def extract_nifti_frame_rate(df: pd.DataFrame) -> Optional[float]:
    pixdim = _val(df, 'pixdim')
    if pixdim is not None and hasattr(pixdim, '__len__') and len(pixdim) > 4:
        dt = float(pixdim[4])
        if dt > 0:
            return round(1.0 / dt, 2)
    return None


def extract_nifti_dimension(df: pd.DataFrame) -> Optional[int]:
    dim = _val(df, 'dim')
    return int(dim[1]) if dim is not None and hasattr(dim, '__len__') and len(dim) > 1 else None
