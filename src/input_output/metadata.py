import numpy as np

from loguru import logger
from PyQt6.QtWidgets import (
    QMainWindow,
    QInputDialog,
    QLineEdit,
    QTableWidgetItem,
)
from PyQt6.QtCore import Qt


class MetadataWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.table = main_window.metadata_table
        self.setWindowTitle('Metadata')

        self.fitToTable()
        self.setCentralWidget(self.table)

    def fitToTable(self):
        x = sum([self.table.columnWidth(i) for i in range(self.table.columnCount())])
        y = sum([self.table.rowHeight(i) for i in range(self.table.rowCount())])
        self.setFixedSize(x, y)


def parse_dicom(main_window):
    modality = main_window.dicom['Modality'].value
    main_window.metadata['modality'] = modality
    if modality == 'OCT':
        parse_ivus_oct(main_window)
    else:
        parse_ivus(main_window)


def parse_ivus(main_window):
    """Parses DICOM metadata for IVUS"""
    if len(main_window.dicom.PatientName.encode('ascii')) > 0:
        patient_name = main_window.dicom.PatientName.original_string.decode('utf-8')
    else:
        patient_name = 'Unknown'

    if len(main_window.dicom.PatientBirthDate) > 0:
        birth_date = main_window.dicom.PatientBirthDate
    else:
        birth_date = 'Unknown'

    if len(main_window.dicom.PatientSex) > 0:
        gender = main_window.dicom.PatientSex
    else:
        gender = 'Unknown'

    if main_window.dicom.get('IVUSPullbackRate'):
        pullback_rate = float(main_window.dicom.IVUSPullbackRate)
    # check Boston private tag
    elif main_window.dicom.get(0x000B1001):
        pullback_rate = float(main_window.dicom[0x000B1001].value)
    else:
        pullback_rate, _ = QInputDialog.getText(
            main_window,
            'Pullback Speed',
            'No pullback speed found, please enter pullback speed (mm/s)',
            QLineEdit.EchoMode.Normal,
            '0.5',
        )
        pullback_rate = float(pullback_rate)

    main_window.metadata['pullback_speed'] = pullback_rate

    if main_window.dicom.get('FrameTimeVector'):
        frame_time_vector = main_window.dicom.get('FrameTimeVector')
        frame_time_vector = [float(frame) for frame in frame_time_vector]
        pullback_time = np.cumsum(frame_time_vector) / 1000  # assume in ms
        pullback_length = pullback_time * float(pullback_rate)
    else:
        pullback_length = np.zeros((main_window.images.shape[0],))

    main_window.metadata['pullback_length'] = pullback_length

    if main_window.dicom.get('SequenceOfUltrasoundRegions'):
        if main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalUnitsXDirection == 3:
            # pixels are in cm, convert to mm
            resolution = main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX * 10
        else:
            # assume mm
            resolution = main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    elif main_window.dicom.get('PixelSpacing'):
        resolution = float(main_window.dicom.PixelSpacing[0])
    else:
        resolution, _ = QInputDialog.getText(
            main_window,
            'Pixel Spacing',
            'No pixel spacing info found, please enter pixel spacing (mm)',
            QLineEdit.EchoMode.Normal,
            '',
        )
        resolution = float(resolution)

    main_window.metadata['resolution'] = resolution

    if main_window.dicom.get('Rows'):
        rows = main_window.dicom.Rows
    else:
        rows = main_window.images.shape[1]

    main_window.metadata['dimension'] = rows

    if main_window.dicom.get('Manufacturer'):
        manufacturer = main_window.dicom.Manufacturer
    else:
        manufacturer = 'Unknown'

    if main_window.dicom.get('ManufacturerModelName'):
        model = main_window.dicom.ManufacturerModelName
    else:
        model = 'Unknown'

    if main_window.dicom.get('IVUSPullbackStartFrameNumber'):
        pullback_start_frame = main_window.dicom.IVUSPullbackStartFrameNumber
    else:
        pullback_start_frame, _ = QInputDialog.getText(
            main_window,
            'Pullback Start Frame',
            'No pullback start frame found, please enter the start frame number',
            QLineEdit.EchoMode.Normal,
            '0',
        )
        pullback_start_frame = int(pullback_start_frame)

    main_window.metadata['pullback_start_frame'] = pullback_start_frame
    main_window.metadata['frame_rate'] = main_window.dicom.get('CineRate')

    main_window.metadata_table.setRowCount(9)
    main_window.metadata_table.setColumnCount(2)
    main_window.metadata_table.setItem(0, 0, QTableWidgetItem('Patient Name'))
    main_window.metadata_table.setItem(0, 1, QTableWidgetItem(patient_name))
    main_window.metadata_table.setItem(1, 0, QTableWidgetItem('Date of Birth'))
    main_window.metadata_table.setItem(1, 1, QTableWidgetItem(birth_date))
    main_window.metadata_table.setItem(2, 0, QTableWidgetItem('Gender'))
    main_window.metadata_table.setItem(2, 1, QTableWidgetItem(gender))
    main_window.metadata_table.setItem(3, 0, QTableWidgetItem('Pullback Speed'))
    main_window.metadata_table.setItem(3, 1, QTableWidgetItem(str(pullback_rate)))
    main_window.metadata_table.setItem(4, 0, QTableWidgetItem('Resolution (mm)'))
    main_window.metadata_table.setItem(4, 1, QTableWidgetItem(str(main_window.metadata['resolution'])))
    main_window.metadata_table.setItem(5, 0, QTableWidgetItem('Dimensions'))
    main_window.metadata_table.setItem(5, 1, QTableWidgetItem(str(rows)))
    main_window.metadata_table.setItem(6, 0, QTableWidgetItem('Manufacturer'))
    main_window.metadata_table.setItem(6, 1, QTableWidgetItem(manufacturer))
    main_window.metadata_table.setItem(7, 0, QTableWidgetItem('Model'))
    main_window.metadata_table.setItem(7, 1, QTableWidgetItem((model)))
    main_window.metadata_table.setItem(8, 0, QTableWidgetItem('Pullback Start Frame'))
    main_window.metadata_table.setItem(8, 1, QTableWidgetItem(str(main_window.metadata['pullback_start_frame'])))

    main_window.metadata_table.horizontalHeader().hide()
    main_window.metadata_table.verticalHeader().hide()
    main_window.metadata_table.resizeColumnsToContents()
    main_window.metadata_table.resizeRowsToContents()
    main_window.metadata_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    main_window.metadata_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


def parse_ivus_oct(main_window):
    """Parses DICOM metadata for IVUS+OCT modality"""
    ds = main_window.dicom

    patient_name = str(ds.get('PatientName', 'Unknown'))
    birth_date = str(ds.get('PatientBirthDate', 'Unknown'))
    gender = str(ds.get('PatientSex', 'Unknown'))

    modality = ds.get('Modality', 'Unknown')
    manufacturer = ds.get('Manufacturer', 'Unknown')
    model = ds.get('ManufacturerModelName', 'Unknown')
    rows = ds.get('Rows', main_window.images.shape[1])

    main_window.metadata['dimension'] = rows

    # Pullback Rate (mm/s)
    if ds.get('IVUSPullbackRate'):
        pullback_rate = float(ds.IVUSPullbackRate)
    elif ds.get(0x000B1001):  # Boston Private Tag
        pullback_rate = float(ds[0x000B1001].value)
    else:
        val, ok = QInputDialog.getText(
            main_window,
            'Pullback Speed',
            f'No speed found for {modality}, enter mm/s:',
            QLineEdit.EchoMode.Normal,
            '0.5',
        )
        pullback_rate = float(val) if ok else 0.5

    main_window.metadata['pullback_speed'] = pullback_rate

    num_frames = main_window.images.shape[0]

    # Pullback length
    if ds.get((0x0018, 0x1063)):
        frame_time_ms = float(ds[0x0018, 0x1063].value)
    elif ds.get('FrameTime'):
        frame_time_ms = float(ds.FrameTime)
    else:
        val, ok = QInputDialog.getText(
            main_window, 'Frame Time', 'No frame time found, enter frame time (ms):', QLineEdit.EchoMode.Normal, '33.3'
        )
        frame_time_ms = float(val) if ok else 33.3
    duration_s = (num_frames * frame_time_ms) / 1000

    main_window.metadata['pullback_length'] = pullback_rate * duration_s

    # Frame rate
    frame_rate = num_frames / duration_s
    # Abbott OPTIS stores FrameTime as ~100ms regardless of actual rate.
    # Derive correct fps from pullback speed per FCC spec (FCC-ID sb6c408650):
    #   36/18 mm/s → 180 fps (standard); 10/20/25 mm/s → 100 fps (C7 Dragonfly)
    if 'abbott' in str(manufacturer).lower() and 'optis' in str(model).lower():
        if abs(pullback_rate - 36.0) < 1 or abs(pullback_rate - 18.0) < 1:
            frame_rate = 180.0
        else:
            frame_rate = 100.0
    main_window.metadata['frame_rate'] = frame_rate

    # Resolution (Pixel Spacing)
    if ds.get('SequenceOfUltrasoundRegions'):
        region = ds.SequenceOfUltrasoundRegions[0]
        unit_multiplier = 10 if region.PhysicalUnitsXDirection == 3 else 1
        resolution = float(region.PhysicalDeltaX) * unit_multiplier
    elif ds.get('PixelSpacing'):
        resolution = float(ds.PixelSpacing[0])
    else:
        val, ok = QInputDialog.getText(
            main_window, 'Pixel Spacing', 'Enter pixel spacing (mm):', QLineEdit.EchoMode.Normal, '0.01'
        )
        resolution = float(val) if ok else 0.01

    main_window.metadata['resolution'] = resolution

    if ds.get('IVUSPullbackStartFrameNumber'):
        start_frame = int(ds.IVUSPullbackStartFrameNumber)
    else:
        start_frame = 0

    main_window.metadata['pullback_start_frame'] = start_frame

    # Update UI Table
    metadata_items = [
        ('Modality', modality),
        ('Patient Name', patient_name),
        ('Date of Birth', birth_date),
        ('Gender', gender),
        ('Pullback Speed', f"{pullback_rate} mm/s"),
        ('Resolution', f"{resolution:.4f} mm"),
        ('Dimensions', f"{rows}x{ds.get('Columns', rows)}"),
        ('Manufacturer', f"{manufacturer} ({model})"),
        ('Start Frame', str(start_frame)),
    ]

    main_window.metadata_table.setRowCount(len(metadata_items))
    main_window.metadata_table.setColumnCount(2)

    for i, (label, value) in enumerate(metadata_items):
        main_window.metadata_table.setItem(i, 0, QTableWidgetItem(label))
        main_window.metadata_table.setItem(i, 1, QTableWidgetItem(str(value)))

    main_window.metadata_table.horizontalHeader().hide()
    main_window.metadata_table.verticalHeader().hide()
    main_window.metadata_table.resizeColumnsToContents()
    main_window.metadata_table.resizeRowsToContents()