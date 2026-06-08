import pytest
import sys
import os
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def simple_config():
    # Minimal config used by Master.__init__ (it reads config.save.autosave_interval)
    return SimpleNamespace(save=SimpleNamespace(autosave_interval=1000))


@pytest.fixture(scope='session')
def qt_app():
    """Provide QApplication instance for the test session"""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_main_window():
    """Create a comprehensive mock main window"""
    mock_win = Mock()

    # Basic attributes
    mock_win.image_displayed = True
    mock_win.metadata = {'num_frames': 100, 'frame_rate': 30, 'resolution': 0.1}
    mock_win.gated_frames_dia = [10, 20, 30]
    mock_win.gated_frames_sys = [15, 25, 35]
    mock_win.hide_contours = False
    mock_win.hide_special_points = False
    mock_win.colormap_enabled = False
    mock_win.filter = None

    # Mock UI elements
    mock_win.diastolic_frame_box = Mock()
    mock_win.systolic_frame_box = Mock()
    mock_win.hide_contours_box = Mock()
    mock_win.hide_special_points_box = Mock()
    mock_win.longitudinal_view = Mock()
    mock_win.small_display = Mock()
    mock_win.display = Mock()
    mock_win.display_slider = Mock()

    # Mock communications
    mock_win.display_frame_comms = Mock()
    mock_win.display_frame_comms.updateBW = Mock()

    # Mock data structure
    mock_win.data = {
        'phases': ['-'] * 100,
        'measures': [[None, None] for _ in range(100)],
        'measure_lengths': [[None, None] for _ in range(100)],
        'reference': [None] * 100,
    }

    # Mock style
    mock_style = Mock()
    mock_icon = Mock()
    mock_style.standardIcon.return_value = mock_icon
    mock_win.style.return_value = mock_style

    return mock_win
