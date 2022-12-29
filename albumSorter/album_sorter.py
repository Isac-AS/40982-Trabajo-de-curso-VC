import logging
from file_checker import check_files
from ui.gui import AlbumSorterUi
from tools.preference_file_handler import PreferenceFileHandler

if __name__ == '__main__':
    if not check_files():
        logging.critical("Exiting program due to file checking error.")
        exit(1)
    
    app = AlbumSorterUi()
    app.start()