from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
from cellpose import models as cellpose_models
from skimage import measure
import cv2
import time
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import svgwrite
from ui_mainwindow import Ui_OpenCVWindow
from plot_window import PlotWindow
from cat_model import SimpleCNN, load_and_prepare_data, get_data_transforms


class ApplicationLogic(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_OpenCVWindow()
        self.ui.setupUi(self)

        # Initialize variables
        self.current_diameter = int(self.ui.diameterEdit.text())
        self.cv_image = None  # OpenCV format image (BGR)
        self.display_image_rgb = None  # RGB image for display (original loaded or with detections)
        self.selection_overlay_image = None  # Image with temporary selections (CV format)
        self.detected_centroids = []
        self.selected_points_for_training = []
        self.is_selection_mode = False
        self.plot_window_instance = None  # To keep a reference to the plot window
        self.min_pos_gray = None

        # --- New variables for zoom and pan ---
        self.base_pixmap_for_display = QPixmap()  # The full, original pixmap currently being shown
        self.scale_factor = 1.0
        self.pan_offset = QtCore.QPointF(0.0, 0.0)  # Using QPointF for precision
        self.is_panning = False
        self.last_mouse_pos = QtCore.QPoint()
        # --- End of new variables ---

        self.connect_signals()
        self.update_button_states()
        self.ui.imageLabel.setMouseTracking(True)  # Enable mouse tracking for cursor changes etc.

    def connect_signals(self):
        self.ui.openButton.clicked.connect(self.load_image_handler)
        self.ui.detdropButton.clicked.connect(self.detect_microreactors_handler)
        self.ui.diameterEdit.textChanged.connect(self.update_diameter_handler)
        self.ui.calibrateButton.clicked.connect(self.calibrate_diameter_handler)

        self.ui.selectButton.clicked.connect(self.start_selection_handler)
        self.ui.confirmButton.clicked.connect(self.confirm_selection_handler)
        self.ui.cancelButton.clicked.connect(self.cancel_selection_handler)

        self.ui.trainButton.clicked.connect(self.train_model_handler)
        self.ui.predictButton.clicked.connect(self.predict_centroids_handler)
        self.ui.clearButton.clicked.connect(self.clear_dataset_handler)
        self.ui.plotButton.clicked.connect(self.plot_histogram_handler)
        self.ui.saveSvgButton.clicked.connect(self.save_svg_handler)

        # Connect mouse and wheel events for image label for zoom/pan
        # This direct assignment works but subclassing QLabel or using an event filter is more robust.
        self.ui.imageLabel.wheelEvent = self.image_label_wheel_event
        self.ui.imageLabel.mousePressEvent = self.image_label_mouse_press_router
        self.ui.imageLabel.mouseMoveEvent = self.image_label_mouse_move_event
        self.ui.imageLabel.mouseReleaseEvent = self.image_label_mouse_release_event

    def update_button_states(self):
        image_loaded = self.cv_image is not None
        centroids_detected = bool(self.detected_centroids)
        model_exists = os.path.exists('cat_model.pth')

        self.ui.detdropButton.setEnabled(image_loaded)
        self.ui.calibrateButton.setEnabled(image_loaded)
        self.ui.selectButton.setEnabled(image_loaded and centroids_detected)  # Enable only if centroids are there
        self.ui.predictButton.setEnabled(image_loaded and centroids_detected and model_exists)
        self.ui.plotButton.setEnabled(image_loaded and centroids_detected)
        self.ui.confirmButton.setEnabled(self.is_selection_mode)
        self.ui.cancelButton.setEnabled(self.is_selection_mode)
        pos_dir = 'pos'
        neg_dir = 'neg'
        dataset_exists = (os.path.exists(pos_dir) and any(f.endswith('.png') for f in os.listdir(pos_dir))) or \
                         (os.path.exists(neg_dir) and any(f.endswith('.png') for f in os.listdir(neg_dir)))
        self.ui.clearButton.setEnabled(dataset_exists)
        self.ui.trainButton.setEnabled(dataset_exists)
        self.ui.saveSvgButton.setEnabled(image_loaded and centroids_detected and model_exists)

    def _cv_to_qpixmap(self, cv_img_bgr):
        """Converts a BGR OpenCV image to QPixmap."""
        if cv_img_bgr is None:
            return QPixmap()

        if len(cv_img_bgr.shape) == 2:  # Grayscale image
            height, width = cv_img_bgr.shape
            bytes_per_line = width
            q_img = QImage(cv_img_bgr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            if not cv_img_bgr.flags['C_CONTIGUOUS']:
                cv_img_bgr = np.ascontiguousarray(cv_img_bgr)
            rgb_image = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _update_base_pixmap_from_cv(self, cv_img_for_display):
        """Updates self.base_pixmap_for_display from the given CV image."""
        self.base_pixmap_for_display = self._cv_to_qpixmap(cv_img_for_display)

    def reset_view(self):
        """Resets zoom and pan to fit the image to the label and updates display."""
        if self.base_pixmap_for_display.isNull():
            self.ui.imageLabel.setPixmap(QPixmap())
            self.ui.imageLabel.setText("No image to display.")
            self.scale_factor = 1.0
            self.pan_offset = QtCore.QPointF(0.0, 0.0)
            return

        label_size = self.ui.imageLabel.size()
        pixmap_size = self.base_pixmap_for_display.size()

        if pixmap_size.width() == 0 or pixmap_size.height() == 0 or label_size.width() == 0 or label_size.height() == 0:
            self.scale_factor = 1.0
        else:
            scale_w = label_size.width() / pixmap_size.width()
            scale_h = label_size.height() / pixmap_size.height()
            self.scale_factor = min(scale_w, scale_h)

        self.scale_factor = max(self.scale_factor, 0.01)  # Minimum sensible scale

        scaled_w = pixmap_size.width() * self.scale_factor
        scaled_h = pixmap_size.height() * self.scale_factor

        self.pan_offset = QtCore.QPointF(
            (label_size.width() - scaled_w) / 2.0,
            (label_size.height() - scaled_h) / 2.0
        )
        self.update_displayed_pixmap()

    def update_displayed_pixmap(self):
        """Renders self.base_pixmap_for_display onto the imageLabel
           according to self.scale_factor and self.pan_offset.
        """
        if self.base_pixmap_for_display.isNull():
            self.ui.imageLabel.setPixmap(QPixmap())
            self.ui.imageLabel.setText("No image to display.")
            return

        label_size = self.ui.imageLabel.size()
        if label_size.width() == 0 or label_size.height() == 0:  # Label not ready
            self.ui.imageLabel.setPixmap(QPixmap())
            return

        display_pixmap = QPixmap(label_size)
        display_pixmap.fill(self.palette().color(QtGui.QPalette.Background))  # Fill with widget background

        painter = QtGui.QPainter(display_pixmap)

        transform = QtGui.QTransform()
        transform.translate(self.pan_offset.x(), self.pan_offset.y())
        transform.scale(self.scale_factor, self.scale_factor)
        painter.setTransform(transform)

        painter.drawPixmap(QtCore.QPointF(0, 0), self.base_pixmap_for_display)

        painter.end()
        self.ui.imageLabel.setText("")  # Clear any "No image" text
        self.ui.imageLabel.setPixmap(display_pixmap)

    def load_image_handler(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_name:
            self.cv_image = cv2.imread(file_name)
            if self.cv_image is None:
                QMessageBox.warning(self, "Image Load Failed", f"Unable to load image: {file_name}")
                self.ui.statusLabel.setText(f'Status: Image load failed {file_name}')
                self.base_pixmap_for_display = QPixmap()  # Clear pixmap
                self.reset_view()  # Update display to show "No image"
                return

            self.display_image_rgb = self.cv_image.copy()
            self.selection_overlay_image = self.cv_image.copy()

            self._update_base_pixmap_from_cv(self.display_image_rgb)
            self.reset_view()  # This will also call update_displayed_pixmap

            self.ui.statusLabel.setText(f'Status: Image loaded ({os.path.basename(file_name)})')
            self.detected_centroids = []
            self.selected_points_for_training = []
            self.is_selection_mode = False  # Ensure selection mode is off on new image
            self.ui.resultLabel.setText('Result: ')
            self.update_button_states()

    def detect_microreactors_handler(self):
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        self.ui.statusLabel.setText('Status: Detecting microreactors...')
        QtWidgets.QApplication.processEvents()
        start_time = time.time()

        try:
            model = cellpose_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model='unet_ywd')
            masks, flows, styles = model.eval(self.cv_image, diameter=self.current_diameter, channels=[0, 0])

            props = measure.regionprops(masks)
            self.detected_centroids = []
            self.display_image_rgb = self.cv_image.copy()  # Start with fresh base CV image

            for prop in props:
                centroid_y, centroid_x = prop.centroid
                centroid = (int(centroid_x), int(centroid_y))
                self.detected_centroids.append(centroid)
                cv2.circle(self.display_image_rgb, centroid, 1, (0, 255, 255), -1)
                cv2.circle(self.display_image_rgb, centroid, self.current_diameter // 2, (255, 0, 0), 2)

            self._update_base_pixmap_from_cv(self.display_image_rgb)  # Update base pixmap
            self.update_displayed_pixmap()  # Refresh view with current zoom/pan

            self.selection_overlay_image = self.display_image_rgb.copy()

            end_time = time.time()
            self.ui.statusLabel.setText(
                f'Status:Microreactor detection completed, time taken {end_time - start_time:.2f}s. {len(self.detected_centroids)} microreactors detected.')
        except Exception as e:
            self.ui.statusLabel.setText(f'Status: Microreactor detection failed: {e}')
            QMessageBox.critical(self, "Detection Error", f"Error occurred during detection: {e}")
        finally:
            self.update_button_states()

    # ... (calibrate_diameter_handler, update_diameter_handler - no changes needed here for zoom/pan)
    def calibrate_diameter_handler(self):
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        self.ui.statusLabel.setText('Status: Calibrating diameter...')
        QtWidgets.QApplication.processEvents()
        try:
            model = cellpose_models.Cellpose(gpu=torch.cuda.is_available(), model_type='cyto2')
            h, w = self.cv_image.shape[:2]
            crop_size = 200
            if h < crop_size or w < crop_size:
                QMessageBox.information(self, "Image Too Small",
                                        "Image size is smaller than the minimum required for calibration (200x200).")
                self.ui.statusLabel.setText('Status: Image too small to calibrate.')
                return

            t_image = self.cv_image[h // 2 - crop_size // 2: h // 2 + crop_size // 2,
                      w // 2 - crop_size // 2: w // 2 + crop_size // 2]
            masks, _, _, diams = model.eval(t_image, diameter=None, channels=[0, 0])

            avg_diam_found = False
            avg_diam = self.current_diameter  # Default to current if not found

            if diams is not None:
                if isinstance(diams, (float, np.float32, np.float64)):  # Cellpose 2.0+ often returns single float
                    avg_diam = int(diams)
                    avg_diam_found = True
                elif isinstance(diams, (np.ndarray, list)) and len(masks.unique()) > 1:
                    props = measure.regionprops(masks)
                    if props:
                        diameters = [prop.equivalent_diameter for prop in props if prop.equivalent_diameter > 0]
                        if diameters:
                            avg_diam = int(np.mean(diameters))
                            avg_diam_found = True

            if avg_diam_found:
                self.ui.diameterEdit.setText(str(avg_diam))
                self.current_diameter = avg_diam
                self.ui.statusLabel.setText(f'Status: Diameter calibration completed. Estimated diameter: {avg_diam}px')
            else:
                self.ui.statusLabel.setText('Status: Failed to estimate diameter in calibration area.')

        except Exception as e:
            self.ui.statusLabel.setText(f'Status: Diameter calibration failed: {e}')
            QMessageBox.critical(self, "Calibration Error", f"An error occurred during calibration: {e}")
        finally:
            self.update_button_states()

    def update_diameter_handler(self, value):
        try:
            val = int(value)
            if 10 <= val <= 300:  # Assuming QIntValidator is set on UI for this range
                self.current_diameter = val
            # else: QIntValidator should handle feedback or correction
        except ValueError:
            self.ui.diameterEdit.setText(str(self.current_diameter))  # Revert if invalid

    def start_selection_handler(self):
        if not self.detected_centroids:
            QMessageBox.information(self, "No Microreactors", "Please detect microreactors before starting selection.")
            return
        self.is_selection_mode = True
        self.selected_points_for_training = []  # Clear previous selections

        self.selection_overlay_image = self.cv_image.copy()
        self._update_base_pixmap_from_cv(self.selection_overlay_image)
        self.update_displayed_pixmap()

        self.ui.statusLabel.setText("Status: Selection mode activated. Please click on the image to annotate.")
        self.update_button_states()

    def handle_point_selection(self, event: QtGui.QMouseEvent):
        if not self.is_selection_mode or self.base_pixmap_for_display.isNull() or not self.detected_centroids or self.cv_image is None:
            return

        try:
            label_click_pos = event.pos()
            label_click_pos = QtCore.QPointF(label_click_pos)

            original_image_x = (label_click_pos.x() - self.pan_offset.x()) / self.scale_factor
            original_image_y = (label_click_pos.y() - self.pan_offset.y()) / self.scale_factor

            original_image_x_int = int(original_image_x)
            original_image_y_int = int(original_image_y)

            if not (0 <= original_image_x_int < self.cv_image.shape[1] and 0 <= original_image_y_int <
                    self.cv_image.shape[0]):
                print(f"The coordinates are out of range: ({original_image_x_int}, {original_image_y_int})")
                return

            nearest_centroid = min(
                self.detected_centroids,
                key=lambda c: (c[0] - original_image_x_int) ** 2 + (c[1] - original_image_y_int) ** 2
            )
            dist_sq = (nearest_centroid[0] - original_image_x_int) ** 2 + (
                    nearest_centroid[1] - original_image_y_int) ** 2

            if dist_sq < (self.current_diameter // 2) ** 2:
                if nearest_centroid not in self.selected_points_for_training:
                    self.selected_points_for_training.append(nearest_centroid)
                    cv2.circle(self.selection_overlay_image, nearest_centroid, self.current_diameter // 2, (0, 0, 255),
                               -1)
                else:
                    self.selected_points_for_training.remove(nearest_centroid)
                    self.selection_overlay_image = self.display_image_rgb.copy()
                    for pt in self.selected_points_for_training:
                        cv2.circle(self.selection_overlay_image, pt, self.current_diameter // 2, (0, 0, 255), -1)

                self._update_base_pixmap_from_cv(self.selection_overlay_image)
                self.update_displayed_pixmap()
        except Exception as e:
            print(f"An error occurred while selecting: {e}")

    def confirm_selection_handler(self):
        if not self.selected_points_for_training:
            QMessageBox.information(self, "No Selection", "Please select at least one microreactor first.")
            return

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Select Annotation Type")
        msg_box.setText("Please choose how to annotate the selected microreactors:")
        pos_button = msg_box.addButton('Mark as Positive', QMessageBox.AcceptRole)
        neg_button = msg_box.addButton('Mark as Negative', QMessageBox.RejectRole)
        cancel_mb_button = msg_box.addButton('Cancel Annotation', QMessageBox.DestructiveRole)
        msg_box.exec_()

        folder = None
        if msg_box.clickedButton() == pos_button:
            folder = 'pos'
        elif msg_box.clickedButton() == neg_button:
            folder = 'neg'
        else:
            self.ui.statusLabel.setText("Status: Annotation cancelled.")
            # No visual change to image needed if cancelling the dialog
            return

        if not os.path.exists(folder):
            os.makedirs(folder)
        crop_radius = self.current_diameter // 4
        saved_count = 0
        for i, centroid in enumerate(self.selected_points_for_training):
            cx, cy = centroid
            min_x = max(0, cx - crop_radius)
            max_x = min(self.cv_image.shape[1], cx + crop_radius)
            min_y = max(0, cy - crop_radius)
            max_y = min(self.cv_image.shape[0], cy + crop_radius)
            roi = self.cv_image[min_y:max_y, min_x:max_x]
            if roi.size == 0:
                continue
            timestamp = int(time.time() * 1000)
            filename = os.path.join(folder, f'microreactor_{timestamp}_{i}.png')
            cv2.imwrite(filename, roi)
            saved_count += 1

        self.ui.statusLabel.setText(
            f'Status: {saved_count} microreactors have been annotated and saved to the "{folder}" folder.')
        self.is_selection_mode = False
        self.selected_points_for_training = []  # Clear selections

        # Revert display to image with detections (no selection highlights)
        self._update_base_pixmap_from_cv(self.display_image_rgb)
        self.update_displayed_pixmap()
        self.update_button_states()

    def cancel_selection_handler(self):
        self.is_selection_mode = False
        self.selected_points_for_training = []  # Clear selections

        # Revert display to image with detections (no selection highlights)
        self._update_base_pixmap_from_cv(self.display_image_rgb)
        self.update_displayed_pixmap()

        self.ui.statusLabel.setText("Status: Select Canceled.")
        self.update_button_states()

    # --- New Event Handlers for Image Label ---
    def image_label_wheel_event(self, event: QtGui.QWheelEvent):
        if self.base_pixmap_for_display.isNull():
            event.ignore()
            return

        delta = event.angleDelta().y()
        # More sensitive zoom factor for finer control
        zoom_increment = 1.05 if delta > 0 else 1 / 1.05

        old_scale = self.scale_factor
        self.scale_factor *= zoom_increment
        # Min/max scale limits
        self.scale_factor = max(0.05, min(self.scale_factor, 20.0))  # Adjust as needed

        mouse_pos_label = event.posF()  # QPointF for precision

        # Point on original image corresponding to mouse cursor before zoom
        # P_orig_x = (mouse_pos_label.x() - self.pan_offset.x()) / old_scale
        # P_orig_y = (mouse_pos_label.y() - self.pan_offset.y()) / old_scale
        # To avoid issues if old_scale is zero or extremely small (though limited by max earlier)
        if abs(old_scale) < 1e-6:  # effectively zero
            orig_x = 0
            orig_y = 0
        else:
            orig_x = (mouse_pos_label.x() - self.pan_offset.x()) / old_scale
            orig_y = (mouse_pos_label.y() - self.pan_offset.y()) / old_scale

        # New pan to keep P_orig at mouse_pos_label: Pan_new = P_label - P_orig * S_new
        self.pan_offset.setX(mouse_pos_label.x() - orig_x * self.scale_factor)
        self.pan_offset.setY(mouse_pos_label.y() - orig_y * self.scale_factor)

        self.update_displayed_pixmap()
        event.accept()

    def image_label_mouse_press_router(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MiddleButton:
            if not self.base_pixmap_for_display.isNull():
                self.is_panning = True
                self.last_mouse_pos = event.pos()
                self.ui.imageLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
            else:
                event.ignore()
        elif event.button() == QtCore.Qt.LeftButton and self.is_selection_mode:
            self.handle_point_selection(event)  # Use the modified selection handler
            event.accept()
        else:
            event.ignore()  # Allow propagation if not handled

    def image_label_mouse_move_event(self, event: QtGui.QMouseEvent):
        if self.is_panning and (event.buttons() & QtCore.Qt.MiddleButton):
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QtCore.QPointF(delta)  # Add as QPointF
            self.last_mouse_pos = event.pos()
            self.update_displayed_pixmap()
            event.accept()
        else:
            event.ignore()

    def image_label_mouse_release_event(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MiddleButton and self.is_panning:
            self.is_panning = False
            self.ui.imageLabel.setCursor(QtCore.Qt.ArrowCursor)  # Or OpenHandCursor
            event.accept()
        else:
            event.ignore()

    # --- End of New Event Handlers ---

    # ... (train_model_handler, predict_centroids_handler, clear_dataset_handler, plot_histogram_handler - no changes needed for zoom/pan)

    def train_model_handler(self):
        self.ui.statusLabel.setText('Status: Preparing training data...')
        QtWidgets.QApplication.processEvents()

        transform = get_data_transforms()
        dataset, image_paths = load_and_prepare_data(transform=transform)

        if dataset is None or not image_paths:
            QMessageBox.warning(self, "No Training Data", "No images found in the 'pos' or 'neg' folders for training.")
            self.ui.statusLabel.setText('Status: No training data, training aborted.')
            self.update_button_states()
            return
        if len(dataset) < 10:
            reply = QMessageBox.question(self, "Insufficient Data",
                                         f"Only {len(dataset)} training samples found. Training may not be effective. Continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self.ui.statusLabel.setText('Status: Training cancelled due to insufficient data.')
                return

        input_size = self.current_diameter // 2
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        device = torch.device("cpu")
        model = SimpleCNN(input_size=input_size).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.ui.statusLabel.setText(f'Status: Training model on {device}...')
        QtWidgets.QApplication.processEvents()
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            self.ui.statusLabel.setText(f'Status: Training... Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            QtWidgets.QApplication.processEvents()

        model_path = 'cat_model.pth'
        torch.save(model.state_dict(), model_path)
        self.ui.statusLabel.setText(f'Status: Training complete, model saved as {model_path}')
        self.update_button_states()

    def predict_centroids_handler(self):
        if not self.detected_centroids:
            QMessageBox.information(self, "No Microreactors", "Please detect microreactors first.")
            return
        if not os.path.exists('cat_model.pth'):
            QMessageBox.warning(self, "No Model", "Model file 'cat_model.pth' not found. Please train the model first.")
            return

        self.ui.statusLabel.setText('Status: Predicting microreactor classes with AI model...')
        QtWidgets.QApplication.processEvents()

        input_size = self.current_diameter // 2
        device = torch.device("cpu")
        model = SimpleCNN(input_size=input_size).to(device)
        try:
            model.load_state_dict(torch.load('cat_model.pth', map_location=device, weights_only=True))
        except RuntimeError as e:
            QMessageBox.critical(self, "Model Load Error",
                                 f"Error loading model state: {e}\nPlease ensure the model file matches the current model architecture.")
            self.ui.statusLabel.setText('Status: Model loading failed.')
            return
        except AttributeError:  # For older PyTorch without weights_only
            model.load_state_dict(torch.load('cat_model.pth', map_location=device))
        model.eval()
        transform = get_data_transforms()

        # Create a new CV image for displaying predictions
        # Based on the cv_image (original) or display_image_rgb (if it has other persistent drawings like detections)
        prediction_cv_image = self.display_image_rgb.copy()  # Start with image that has detections

        positive_count = 0
        negative_count = 0
        crop_radius = self.current_diameter // 4
        self.min_pos_gray = None  # Initialize minimum positive gray value
        self.prediction_results = []
        with torch.no_grad():
            for centroid in self.detected_centroids:
                cx, cy = centroid
                min_x = max(0, cx - crop_radius)
                max_x = min(self.cv_image.shape[1], cx + crop_radius)
                min_y = max(0, cy - crop_radius)
                max_y = min(self.cv_image.shape[0], cy + crop_radius)

                roi = self.cv_image[min_y:max_y, min_x:max_x]  # Use original cv_image for ROI
                if roi.size == 0: continue
                # Ensure ROI has the correct size
                roi_h, roi_w = roi.shape[:2]
                if roi_h < 2 * crop_radius or roi_w < 2 * crop_radius:
                    roi = cv2.resize(roi, (2 * crop_radius, 2 * crop_radius), interpolation=cv2.INTER_LINEAR)

                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                output = model(roi_tensor)
                _, predicted_class = torch.max(output, 1)

                color = (255, 0, 0)  # Blue for Negative (class 0)
                if predicted_class.item() == 1:  # Positive
                    color = (0, 0, 255)  # Red for Positive (class 1)
                    positive_count += 1
                    # Calculate gray mean
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mean_gray = np.mean(gray_roi)
                    if self.min_pos_gray is None or mean_gray <= self.min_pos_gray:
                        self.min_pos_gray = mean_gray
                    pred_label = 'positive'
                else:
                    negative_count += 1
                    pred_label = 'negative'
                # Draw prediction circles on the prediction_cv_image
                cv2.circle(prediction_cv_image, centroid, self.current_diameter // 2, color, 2)
                self.prediction_results.append({
                    'centroid': centroid,
                    'label': pred_label
                })
        self._update_base_pixmap_from_cv(prediction_cv_image)  # Update base pixmap with prediction drawings
        self.update_displayed_pixmap()  # Refresh view

        total_microreactors = len(self.detected_centroids)
        neg_ratio = negative_count / total_microreactors if total_microreactors > 0 else 0
        pos_ratio = positive_count / total_microreactors if total_microreactors > 0 else 0

        self.ui.statusLabel.setText('Status: AI prediction completed.')
        self.ui.resultLabel.setText(
            f'AI Result: Positive: {positive_count} ({pos_ratio * 100:.2f}%), Negative: {negative_count} ({neg_ratio * 100:.2f}%) / Total: {total_microreactors}')
        self.update_button_states()


    def clear_dataset_handler(self):
        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     "Are you sure you want to delete all training images in the 'pos' and 'neg' folders?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for folder in ['pos', 'neg']:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f'Failed to delete {file_path}. Reason: {e}')
            self.ui.statusLabel.setText("Status: Training dataset has been cleared.")
        else:
            self.ui.statusLabel.setText("Status: Clear operation cancelled.")
        self.update_button_states()


    def save_svg_handler(self):
        if not hasattr(self, 'prediction_results') or not self.prediction_results:
            QtWidgets.QMessageBox.information(self, "No predicted results", "Please make an AI prediction first.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the SVG file", "", "SVG Files (*.svg)")
        if not file_path:
            return

        h, w = self.cv_image.shape[:2]
        dwg = svgwrite.Drawing(file_path, size=(w, h))
        try:
            import io
            import base64
            _, buffer = cv2.imencode('.png', self.cv_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            href = f'data:image/png;base64,{img_base64}'
            dwg.add(dwg.image(href=href, insert=(0, 0), size=(w, h)))
        except Exception as e:
            print(f"Failed to embed the original image: {e}")

        radius = self.current_diameter // 2
        for item in self.prediction_results:
            cx, cy = item['centroid']
            label = item['label']
            color = 'red' if label == 'positive' else 'blue'
            dwg.add(dwg.circle(center=(cx, cy), r=radius, fill='none', stroke=color, stroke_width=2))

        dwg.save()
        self.ui.statusLabel.setText(f"The prediction results are saved as SVG:{file_path}")

    def plot_histogram_handler(self):
        if not self.detected_centroids or self.cv_image is None:
            QMessageBox.information(self, "No Microreactors", "Please load an image and detect microreactors first.")
            return

        # Threshold is taken directly from self.min_pos_gray
        threshold = self.min_pos_gray

        all_centroid_intensities = []
        crop_radius = self.current_diameter // 4
        for centroid in self.detected_centroids:
            cx, cy = centroid
            min_x = max(0, cx - crop_radius)
            max_x = min(self.cv_image.shape[1], cx + crop_radius)
            min_y = max(0, cy - crop_radius)
            max_y = min(self.cv_image.shape[0], cy + crop_radius)
            roi = self.cv_image[min_y:max_y, min_x:max_x]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                all_centroid_intensities.append(np.mean(gray_roi))

        if not all_centroid_intensities:
            QMessageBox.information(self, "No Intensity Data", "Unable to extract centroid intensity of microreactors.")
            return

        if self.plot_window_instance is None or not self.plot_window_instance.isVisible():
            self.plot_window_instance = PlotWindow(self)

        self.plot_window_instance.plot_hist_and_scatter(all_centroid_intensities,
                                                        threshold if threshold is not None else -1)
        self.plot_window_instance.show()
        self.plot_window_instance.activateWindow()

        if threshold is not None:
            below_threshold_count = sum(1 for intensity in all_centroid_intensities if intensity < threshold)
            ratio_below_threshold = below_threshold_count / len(
                all_centroid_intensities) if all_centroid_intensities else 0
            self.ui.statusLabel.setText(
                f"Status: Threshold is the minimum gray value of AI positive: {threshold:.2f}")
            self.ui.resultLabel.setText(
                f'Threshold method: {below_threshold_count} microreactors below threshold ({ratio_below_threshold * 100:.2f}%).')
        else:
            self.ui.resultLabel.setText(f'Threshold method: Threshold not calculated.')

    def resizeEvent(self, event: QtGui.QResizeEvent):
        """Handle window resize to update image display correctly."""
        super().resizeEvent(event)
        # When window resizes, imageLabel might resize.
        # A full reset_view() would lose current zoom/pan state if not desired.
        # Simply calling update_displayed_pixmap() will repaint with current zoom/pan
        # into the new label size. If you want to "refit" the image on resize, call self.reset_view().
        # For now, let's maintain current zoom/pan relative to top-left.
        if hasattr(self, 'base_pixmap_for_display') and not self.base_pixmap_for_display.isNull():
            self.update_displayed_pixmap()
        elif hasattr(self, 'ui'):  # if ui is setup
            self.reset_view()  # If no image, ensure "No image" is centered or label is cleared

    def closeEvent(self, event):
        """Ensure child windows are closed when main window closes."""
        if self.plot_window_instance and self.plot_window_instance.isVisible():
            self.plot_window_instance.close()
        super().closeEvent(event)
