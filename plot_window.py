from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
class PlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setWindowTitle('Droplet Intensity')
        self.setGeometry(150, 150, 1600, 800)
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.92, wspace=0.25)

    def plot_hist_and_scatter(self, centroid_intensities, threshold):
        self.ax1.clear()
        self.ax2.clear()

        if centroid_intensities:
            self.ax1.hist(centroid_intensities, bins=50, range=(0, 255), color='gray', alpha=0.7,
                          label='Intensity Distribution')
            if threshold is not None:
                self.ax1.axvline(x=threshold, color='r', linestyle='--')
                self.ax1.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
            self.ax1.set_xlabel('Average intensity of ROI', fontsize=36, fontweight='bold')
            self.ax1.set_ylabel('Number of Microreactors', fontsize=36, fontweight='bold')
            self.ax1.legend(loc='upper right',prop={'size': 24, 'weight': 'bold'})
            self.ax1.tick_params(axis='x', labelsize=28, width=4)
            for label in self.ax1.get_xticklabels():
                label.set_fontweight('bold')
            self.ax1.tick_params(axis='y', labelsize=28, width=4)
            for label in self.ax1.get_yticklabels():
                label.set_fontweight('bold')
            for spine in self.ax1.spines.values():
                spine.set_linewidth(4)

        else:
            self.ax1.text(0.5, 0.5, "No data to display.", horizontalalignment='center', verticalalignment='center')


        if centroid_intensities:
            self.ax2.scatter(range(len(centroid_intensities)), centroid_intensities, c='b', s=10, label='Intensity')
            if threshold is not None:
                self.ax1.axvline(x=threshold, color='r', linestyle='--')
                self.ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
            self.ax2.set_xlabel('Index of microreactors', fontsize=36, fontweight='bold')
            self.ax2.set_ylabel('Average intensity', fontsize=36, fontweight='bold')
            self.ax2.legend(loc='upper right',prop={'size': 12, 'weight': 'bold'})
            self.ax2.tick_params(axis='x', labelsize=28, width=4)
            for label in self.ax2.get_xticklabels():
                label.set_fontweight('bold')
            self.ax2.tick_params(axis='y', labelsize=28, width=4)
            for label in self.ax2.get_yticklabels():
                label.set_fontweight('bold')
            for spine in self.ax2.spines.values():
                spine.set_linewidth(4)
        else:
            self.ax2.text(0.5, 0.5, "No data to display.", horizontalalignment='center', verticalalignment='center',)

        self.canvas.draw()