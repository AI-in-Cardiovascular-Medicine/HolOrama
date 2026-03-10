import os
import sys
import hydra

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import qdarktheme

from omegaconf import DictConfig
from PyQt6.QtWidgets import QApplication

from version import __version__
from gui.gui import Master

@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)
    
    qdarktheme.setup_theme('dark') # switch to auto to recognize system mode
    _window = Master(config)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()