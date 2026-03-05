import os

from loguru import logger
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtCore import QUrl


class VideoPlayer(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle('Video Player')
        self.resize(600, 400)
        self.video_widget = QVideoWidget()
        self.setCentralWidget(self.video_widget)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.mediaStatusChanged.connect(self.media_ended)

    def play(self, video_path, local_file=True):
        if local_file:
            path = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.abspath(os.path.join(path, '..', '..', video_path))
            media_source = QUrl.fromLocalFile(video_path)
        else:
            media_source = QUrl(video_path)
            
        self.show()
        self.media_player.setSource(media_source)
        self.media_player.setPosition(0)
        self.media_player.play()

    def media_ended(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.close()