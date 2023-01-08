from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from tools.preference_file_handler import PreferenceFileHandler
from face_recognition_and_processing.pipeline import compute_sort
from file_checker import get_current_output_subdirectory
import time
import os

class Album_Sorter_Window(QMainWindow):

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Sorter window", "Sorter window"))
        self.infoLabel.setText(_translate("Sorter window", "  Choose folder with images:"))
        self.browseButton.setText(_translate("Sorter window", "  Browse  "))
        self.sortingAction.setText(_translate("Sorter window", "  Compute sort!  "))


    def init_main_frame(self):
        self.setObjectName("Sorter window")
        self.resize(1100, 250)
        # Font declaration
        font = QtGui.QFont()
        # Central widget
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        # Display flex vertical
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        
        # Tag with info
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        # Info label within this layout
        self.infoLabel = QtWidgets.QLabel(self)
        font.setPointSize(26)
        self.infoLabel.setFont(font)
        self.infoLabel.setObjectName("infoLabel")
        # Add the tag to the widget and add the widget to the layout
        self.horizontalLayout.addWidget(self.infoLabel)
        self.verticalLayout.addLayout(self.horizontalLayout)
        
        # Second horizontal layout with browse button
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        # Spacer item 0
        #spacerItem = QtWidgets.QSpacerItem(5, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.horizontalLayout_3.addItem(spacerItem)
        # Line Edit (Text area/field)
        self.pathToBaseFolder = QtWidgets.QLineEdit(self)
        font.setPointSize(22)
        self.pathToBaseFolder.setFont(font)
        self.pathToBaseFolder.setMaxLength(32000)
        self.pathToBaseFolder.setObjectName("pathToBaseFolder")
        self.horizontalLayout_3.addWidget(self.pathToBaseFolder)
        # Spacer item 1
        #spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.horizontalLayout_3.addItem(spacerItem1)
        # Browse button
        self.browseButton = QtWidgets.QPushButton(self)
        font.setPointSize(22)
        self.browseButton.setFont(font)
        self.browseButton.setObjectName("browseButton")
        self.horizontalLayout_3.addWidget(self.browseButton)
        # Spacer item 2
        #spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.horizontalLayout_3.addItem(spacerItem2)
        # Add second horizontal layout to vertical layout
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        # Third horizontal layout
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # Spacer item
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        # Sort button
        self.sortingAction = QtWidgets.QPushButton(self)
        font.setPointSize(24)
        self.sortingAction.setFont(font)
        self.sortingAction.setObjectName("sortingAction")
        self.horizontalLayout_2.addWidget(self.sortingAction)
        # Spacer item
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        # Add third layout to vertical layout
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        
        # Set central widget
        self.setCentralWidget(self.centralwidget)

        # Connect buttons with functions
        self.browseButton.clicked.connect(self.browse_action)
        self.sortingAction.clicked.connect(self.sort_action)

    def __init__(self):
        super().__init__()
        # Init frame components
        self.init_main_frame()
        # Rename them
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # Check for last browsed folder
        self.check_for_base_dir()

        # Set tab order
        self.setTabOrder(self.infoLabel, self.browseButton)
        self.setTabOrder(self.browseButton, self.sortingAction)

    def check_for_base_dir(self):
        """
        This function will attempt to load the last directory computed from the 
        JSON file.
        This extra functionality will attempt to avoid selecting a directory every time.
        """
        base_dir = PreferenceFileHandler.get_base_directory()
        if base_dir != "None":
            self.pathToBaseFolder.setText(base_dir)
    
    # Actions
    def sort_action(self):
        """
        Will attempt to carry out the face recognition and classification algorithm on the selected folder.
        """
        _translate = QtCore.QCoreApplication.translate
        self.sortingAction.setText(_translate("Sorter window", "  In progress ...  "))
        self.sortingAction.repaint()
        self.save_base_directory()
        PreferenceFileHandler.set_output_directory(get_current_output_subdirectory())
        compute_sort()
        os.system(f"start {os.path.abspath(PreferenceFileHandler.get_output_directory())}")
        self.sortingAction.setText(_translate("Sorter window", "  Compute new sort!  "))
        return None

    def browse_action(self):
        """
        Will open a browse interface to select a folder.
        """
        pathName = QtWidgets.QFileDialog.getExistingDirectory(self, QtCore.QDir.rootPath())
        self.pathToBaseFolder.setText(pathName)

    def save_base_directory(self):
        """
        Will save the selected folder on the JSON document.
        """
        PreferenceFileHandler.set_base_directory(self.pathToBaseFolder.text())
