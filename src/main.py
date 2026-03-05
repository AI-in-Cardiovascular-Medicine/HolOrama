import sys
import hydra
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
    Master(config)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()