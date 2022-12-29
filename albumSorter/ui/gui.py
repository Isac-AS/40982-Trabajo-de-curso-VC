import sys
from PyQt5.QtWidgets import QApplication
from ui.sorter_window_ui_0 import Album_Sorter_Window

class AlbumSorterUi():
    def __init__(self) -> None:
        self.app = QApplication(sys.argv)
        self.main_window = Album_Sorter_Window()

    def start(self):
        self.main_window.show()
        self.app.exec()
