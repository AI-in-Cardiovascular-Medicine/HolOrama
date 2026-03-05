from loguru import logger
from PyQt6.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPen
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from scipy.ndimage import gaussian_filter1d

from report.report import report


class ResultsPlot(QMainWindow):
    def __init__(self, main_window, report_data):
        super().__init__(main_window)
        self.main_window = main_window
        self.report_data = report_data
        self.pullback_speed = main_window.metadata.get('pullback_speed', 1)
        self.pullback_start_frame = main_window.metadata.get('pullback_start_frame', 0)
        self.frame_rate = main_window.metadata.get('frame_rate', 30)

        self.setWindowTitle('Results Plot')
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.plot_results()

    def plot_results(self):
        self.scene.clear()
        self.scene.setSceneRect(0, 0, 1000, 1200)

        df = self.prep_data()
        df['phase'] = df['phase'].replace({'D': 'Diastole', 'S': 'Systole'})

        # Create a matplotlib figure with two subplots and increased vertical space
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})

        # Plot lumen area by phase
        min_lumen_area = float('inf')
        min_lumen_area_value = None
        min_lumen_area_distance = None
        min_lumen_area_frame = None

        for phase, group in df.groupby('phase'):
            # Smooth the lumen_area
            smoothed_area = gaussian_filter1d(group['lumen_area'], sigma=2)  # Adjust sigma for smoothing
            ax1.plot(group['distance'], smoothed_area, label=f'Lumen Area - {phase}')
            ax1.scatter(group['distance'], group['lumen_area'], alpha=0.3)

            # Find the minimum lumen area across all phases
            phase_min_lumen_area = group['lumen_area'].min()
            if phase_min_lumen_area < min_lumen_area:
                min_lumen_area = phase_min_lumen_area
                min_lumen_area_value = phase_min_lumen_area
                min_lumen_area_distance = group.loc[group['lumen_area'].idxmin(), 'distance']
                min_lumen_area_frame = group.loc[group['lumen_area'].idxmin(), 'frame']

            # Define colors based on phase
            ostial_color = '#008b8b' if phase == 'Diastole' else '#ff6f00'
            min_area_color = '#0055ff'

            # Highlight the lumen_area at distance 0 for ostial area
            ostial_lumen_area = group.loc[group['distance'] == 0, 'lumen_area'].values
            if len(ostial_lumen_area) > 0:
                ax1.scatter(0, ostial_lumen_area[0], color=ostial_color, zorder=5)
                ax1.text(
                    0,
                    ostial_lumen_area[0],
                    f'{ostial_lumen_area[0]:.2f} ({group.loc[group["distance"] == 0, "frame"].values[0]})',
                    color=ostial_color,
                )

        # Highlight the smallest lumen area with a specific color
        if min_lumen_area_value is not None and min_lumen_area_distance is not None:
            ax1.scatter(min_lumen_area_distance, min_lumen_area_value, color=min_area_color, zorder=5)
            ax1.text(
                min_lumen_area_distance,
                min_lumen_area_value,
                f'{min_lumen_area_value:.2f} ({min_lumen_area_frame})',
                color=min_area_color,
            )

        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Lumen Area (mm²)')
        ax1.set_title('Lumen Area vs Distance by Phase')
        ax1.invert_xaxis()  # Invert the x-axis to start x=0 on the right side
        ax1.legend()

        # Add a second x-axis for frames
        ax1_frames = ax1.twiny()
        ax1_frames.set_xlim(ax1.get_xlim())
        ax1_frames.set_xticks(df['distance'][::5])  # Keep every 5th frame
        ax1_frames.set_xticklabels(df['frame'][::5])
        ax1_frames.set_xlabel('Frames')

        # Plot elliptic ratio by phase
        for phase, group in df.groupby('phase'):
            # Smooth the elliptic_ratio
            smoothed_ratio = gaussian_filter1d(group['elliptic_ratio'], sigma=2)  # Adjust sigma for smoothing
            ax2.plot(group['distance'], smoothed_ratio, label=f'Elliptic Ratio - {phase}')
            ax2.scatter(group['distance'], group['elliptic_ratio'], alpha=0.3)

        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel('Elliptic Ratio')
        ax2.set_title('Elliptic Ratio vs Distance by Phase')
        ax2.invert_xaxis()  # Invert the x-axis to start x=0 on the right side
        ax2.legend()

        # Add a second x-axis for frames
        ax2_frames = ax2.twiny()
        ax2_frames.set_xlim(ax2.get_xlim())
        ax2_frames.set_xticks(df['distance'][::5])  # Keep every 5th frame
        ax2_frames.set_xticklabels(df['frame'][::5])
        ax2_frames.set_xlabel('Frames')

        # Save the plot to a QImage
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = QImage(
            fig.canvas.buffer_rgba(), 
            width, 
            height, 
            QImage.Format.Format_ARGB32
        )

        self.scene.addPixmap(QPixmap.fromImage(img))

    def prep_data(self):
        df = self.report_data[self.report_data['phase'] != '-'].copy()  # Use copy to avoid warnings
        # drop every row with 'frame' < pullback_start_frame. otherwise the distance calculation will be wrong
        df = df[df['frame'] >= self.pullback_start_frame].copy()

        df_dia = df[df['phase'] == 'D'].copy()  # Ensure a copy
        df_sys = df[df['phase'] == 'S'].copy()  # Ensure a copy

        # Calculate distance based on framerate and pullback speed
        df_dia.loc[:, 'distance'] = (df_dia['frame'].max() - df_dia['frame']) / self.frame_rate * self.pullback_speed
        df_sys.loc[:, 'distance'] = (df_sys['frame'].max() - df_sys['frame']) / self.frame_rate * self.pullback_speed

        df = pd.concat([df_dia, df_sys])

        return df

    def closeEvent(self, event):
        self.main_window.results_plot = None
        event.accept()
