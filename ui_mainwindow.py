from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QLineEdit, QScrollArea, QGroupBox, QVBoxLayout, QHBoxLayout

class Ui_OpenCVWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # === Image Loading and Processing Area ===
        self.image_groupbox = QGroupBox("Image Processing")
        self.image_layout = QVBoxLayout()

        self.openButton = QtWidgets.QPushButton("Open Image")
        self.detdropButton = QtWidgets.QPushButton("Detect Microreactors")
        self.detdropButton.setEnabled(False)

        self.diameterLabel = QLabel("Microreactor Diameter (px):")
        self.diameterEdit = QLineEdit()
        self.diameterEdit.setText("60")
        self.diameterEdit.setValidator(QtGui.QIntValidator(10, 300))
        self.calibrateButton = QtWidgets.QPushButton("Calibrate Diameter")

        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(self.diameterLabel)
        diameter_layout.addWidget(self.diameterEdit)
        diameter_layout.addWidget(self.calibrateButton)

        self.image_layout.addWidget(self.openButton)
        self.image_layout.addWidget(self.detdropButton)
        self.image_layout.addLayout(diameter_layout)
        self.image_groupbox.setLayout(self.image_layout)

        # === Microreactor Selection Area ===
        self.selection_groupbox = QGroupBox("Microreactor Selection & Annotation (for Training)")
        self.selection_layout = QVBoxLayout()

        self.selectButton = QtWidgets.QPushButton("Start Selecting Points")
        self.selectButton.setEnabled(False)
        self.confirmButton = QtWidgets.QPushButton("Confirm Annotation")
        self.confirmButton.setEnabled(False)
        self.cancelButton = QtWidgets.QPushButton("Cancel Selection")
        self.cancelButton.setEnabled(False)

        self.selection_layout.addWidget(self.selectButton)
        self.selection_layout.addWidget(self.confirmButton)
        self.selection_layout.addWidget(self.cancelButton)
        self.selection_groupbox.setLayout(self.selection_layout)

        # === Model Training and Prediction Area ===
        self.model_groupbox = QGroupBox("Model Operations")
        self.model_layout = QVBoxLayout()

        self.trainButton = QtWidgets.QPushButton("Train Model")
        self.predictButton = QtWidgets.QPushButton("Predict Positive/Negative")
        self.predictButton.setEnabled(False)
        self.clearButton = QtWidgets.QPushButton("Clear Training Dataset")
        self.saveSvgButton = QtWidgets.QPushButton("Save as an SVG image")
        self.saveSvgButton.setEnabled(False)

        self.model_layout.addWidget(self.trainButton)
        self.model_layout.addWidget(self.predictButton)
        self.model_layout.addWidget(self.clearButton)
        self.model_groupbox.setLayout(self.model_layout)
        self.model_layout.addWidget(self.saveSvgButton)

        # === Plot Histogram and Scatter Area ===
        self.plotButton = QtWidgets.QPushButton("Plot Histogram and Scatter")
        self.plotButton.setEnabled(False)

        # === Image and Result Display Area ===
        self.image_display_groupbox = QGroupBox("Image Display")
        self.image_display_layout = QVBoxLayout()
        self.imageLabel = QLabel("Please open an image first")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(600, 400)
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setWidget(self.imageLabel)
        self.image_display_layout.addWidget(self.image_scroll_area)
        self.image_display_groupbox.setLayout(self.image_display_layout)

        # Status display
        self.statusLabel = QLabel('Status: Ready')
        self.resultLabel = QLabel('Result: ')

        # === Main Layout ===
        self.main_layout = QVBoxLayout(self.centralwidget)

        self.top_controls_layout = QHBoxLayout()
        self.top_controls_layout.addWidget(self.image_groupbox)
        self.top_controls_layout.addWidget(self.selection_groupbox)
        self.top_controls_layout.addWidget(self.model_groupbox)

        self.main_layout.addLayout(self.top_controls_layout)
        self.main_layout.addWidget(self.image_display_groupbox)
        self.main_layout.addWidget(self.plotButton)
        self.main_layout.addWidget(self.statusLabel)
        self.main_layout.addWidget(self.resultLabel)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("OpenCVWindow", "Microreactor-Vision"))
