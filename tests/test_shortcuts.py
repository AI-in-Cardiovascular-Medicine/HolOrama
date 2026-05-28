from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


class TestShortcuts:
    """Test keyboard shortcuts functionality"""

    def create_mock_main_window(self):
        """Create a mock main window for shortcut testing"""
        mock_win = MagicMock()
        mock_win.image_displayed = True
        mock_win.display = MagicMock()
        mock_win.display.frame = 42
        mock_win.display.contour_key.return_value = "lumen"
        mock_win.display_slider = MagicMock()
        mock_win.data = {'phases': ['-'] * 100, 'lumen': [[[] for _ in range(100)], [[] for _ in range(100)]]}
        mock_win.gated_frames_dia = [10, 20, 30]
        mock_win.gated_frames_sys = [15, 25, 35]
        mock_win.diastolic_frame_box = MagicMock()
        mock_win.systolic_frame_box = MagicMock()
        mock_win.contour_based_gating = MagicMock()
        mock_win.hide_contours_box = MagicMock()
        mock_win.hide_special_points_box = MagicMock()
        mock_win.status_bar = MagicMock()
        mock_win.waiting_status = "Ready"
        mock_win.hide_contours = False
        mock_win.hide_special_points = False
        mock_win.filter = None
        mock_win.colormap_enabled = False
        mock_win.longitudinal_view = MagicMock()
        mock_win.small_display = MagicMock()
        mock_win.tmp_contours = {}

        return mock_win

    def test_hide_contours_shortcut(self):
        """Test hide contours shortcut"""
        from gui.shortcuts import hide_contours

        mock_main_window = self.create_mock_main_window()

        hide_contours(mock_main_window)

        mock_main_window.hide_contours_box.setChecked.assert_called_with(
            not mock_main_window.hide_contours_box.isChecked()
        )

    def test_hide_special_points_shortcut(self):
        """Test hide special points shortcut"""
        from gui.shortcuts import hide_special_points

        mock_main_window = self.create_mock_main_window()

        hide_special_points(mock_main_window)

        mock_main_window.hide_special_points_box.setChecked.assert_called_with(
            not mock_main_window.hide_special_points_box.isChecked()
        )

    @patch('time.sleep')
    @patch('PyQt6.QtWidgets.QApplication.processEvents')
    def test_jiggle_frame_shortcut(self, mock_process_events, mock_sleep):
        """Test jiggle frame shortcut"""
        from gui.shortcuts import jiggle_frame

        mock_main_window = self.create_mock_main_window()
        current_frame = 50
        mock_main_window.display_slider.value.return_value = current_frame

        jiggle_frame(mock_main_window)

        expected_calls = [((current_frame + 1,),), ((current_frame,),), ((current_frame - 1,),), ((current_frame,),)]
        assert mock_main_window.display_slider.set_value.call_args_list == expected_calls

    def test_reset_windowing_shortcut(self):
        """Test reset windowing shortcut"""
        from gui.shortcuts import reset_windowing

        mock_main_window = self.create_mock_main_window()

        # Set non-default values
        mock_main_window.display.window_level = 200
        mock_main_window.display.window_width = 300
        mock_main_window.display.initial_window_level = 128
        mock_main_window.display.initial_window_width = 256

        reset_windowing(mock_main_window)

        # Should reset to initial values
        assert mock_main_window.display.window_level == 128
        assert mock_main_window.display.window_width == 256

        mock_main_window.display.display_image.assert_called_once_with(update_image=True)

    def test_toggle_color_shortcut(self):
        """Test toggle color map shortcut"""
        from gui.shortcuts import toggle_color

        mock_main_window = self.create_mock_main_window()
        initial_state = mock_main_window.colormap_enabled

        toggle_color(mock_main_window)

        assert mock_main_window.colormap_enabled == (not initial_state)

        mock_main_window.display.display_image.assert_called_once_with(update_image=True)
