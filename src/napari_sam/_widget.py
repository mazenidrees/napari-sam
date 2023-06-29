import copy
import inspect
import numpy as np
import os
from collections import deque, defaultdict
from enum import Enum
from os.path import join
from pathlib import Path
import urllib.request
import warnings

import napari
import torch
from tqdm import tqdm
from vispy.util.keys import CONTROL
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QComboBox,
    QRadioButton,
    QGroupBox,
    QProgressBar,
    QApplication,
    QScrollArea,
    QLineEdit,
    QCheckBox,
    QListWidget,
)
from collections import Counter
from superqt.utils import qdebounced

from napari_sam._ui_elements import UiElements, AnnotatorMode
from napari_sam.slicer import slicer
from napari_sam.utils import normalize
from segment_anything import (
    SamPredictor,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
)
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator



class SegmentationMode(Enum):
    SEMANTIC = 0
    INSTANCE = 1

class BboxState(Enum):
    CLICK = 0
    DRAG = 1
    RELEASE = 2

SAM_MODELS = {
    "default": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model": build_sam_vit_l},
    "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model": build_sam_vit_b},
    "MedSAM": {"filename": "sam_vit_b_01ec64_medsam.pth", "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth", "model": build_sam_vit_b},
}

class SamWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.ui_elements = UiElements(self.viewer)
        self.setLayout(self.ui_elements.main_layout)

        #### setting up ui callbacks ####
        self.ui_elements.set_external_handler_btn_load_model(self.load_model)
        self.ui_elements.set_external_handler_btn_activate(self.activate, self.deactivate)
    ################################ model loading ################################
    def load_model(self, model_type):
        if not torch.cuda.is_available():
            if not torch.backends.mps.is_available():
                self.device = "cpu"
            else:
                self.device = "mps"
        else:
            self.device = "cuda"
        self.sam_model = SAM_MODELS[model_type]["model"](self.get_weights_path(model_type))

        self.sam_model.to(self.device)
        print("debug 4")
        self.sam_predictor = SamPredictor(self.sam_model)
        print("debug 5")

    def get_weights_path(self, model_type):
        weight_url = SAM_MODELS[model_type]["url"]

        cache_dir = Path.home() / ".cache/napari-segment-anything"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weight_path = cache_dir / SAM_MODELS[model_type]["filename"]

        if not weight_path.exists():
            print("Downloading {} to {} ...".format(weight_url, weight_path))
            self.download_with_progress(weight_url, weight_path)

        return weight_path

    def download_with_progress(self, url, output_file):
        # Open the URL and get the content length
        req = urllib.request.urlopen(url)
        content_length = int(req.headers.get('Content-Length'))

        self.ui_elements.create_progress_bar(int(content_length / 1024), "Downloading model:")

        # Set up the tqdm progress bar
        progress_bar_tqdm = tqdm(total=content_length, unit='B', unit_scale=True, desc="Downloading model")

        # Download the file and update the progress bar
        with open(output_file, 'wb') as f:
            downloaded_bytes = 0
            while True:
                buffer = req.read(8192)
                if not buffer:
                    break
                downloaded_bytes += len(buffer)
                f.write(buffer)
                progress_bar_tqdm.update(len(buffer))

                # Update the progress bar using UiElements method
                self.ui_elements.update_progress_bar(int(downloaded_bytes / 1024))

        self.ui_elements.delete_progress_bar()

        progress_bar_tqdm.close()
        req.close()

    ################################ activating sam ################################
    def activate(self, annotator_mode):
        self.set_layers()
        self.adjust_image_layer_shape()
        self.check_image_dimension()
        self.set_sam_logits()

        if annotator_mode == AnnotatorMode.AUTO:
            self.activate_annotation_mode_auto()

        elif annotator_mode == AnnotatorMode.CLICK:
            self.activate_annotation_mode_click()

    #### preparing
    def set_layers(self):
        self.image_name = self.ui_elements.cb_input_image_selctor.currentText()
        self.image_layer = self.viewer.layers[self.ui_elements.cb_input_image_selctor.currentText()]
        self.label_layer = self.viewer.layers[self.ui_elements.cb_output_label_selctor.currentText()]
        self.label_layer_changes = None

    def adjust_image_layer_shape(self):
        if self.image_layer.ndim == 3:
            # Fixes shape adjustment by napari
            self.image_layer_affine_scale = self.image_layer.affine.scale
            self.image_layer_scale = self.image_layer.scale
            self.image_layer_scale_factor = self.image_layer.scale_factor
            self.label_layer_affine_scale = self.label_layer.affine.scale
            self.label_layer_scale = self.label_layer.scale
            self.label_layer_scale_factor = self.label_layer.scale_factor
            self.image_layer.affine.scale = np.array([1, 1, 1])
            self.image_layer.scale = np.array([1, 1, 1])
            self.image_layer.scale_factor = 1
            self.label_layer.affine.scale = np.array([1, 1, 1])
            self.label_layer.scale = np.array([1, 1, 1])
            self.label_layer.scale_factor = 1
            pos = self.viewer.dims.point
            self.viewer.dims.set_point(0, 0)
            self.viewer.dims.set_point(0, pos[0])
            self.viewer.reset_view()

    def check_image_dimension(self):
        if self.image_layer.ndim != 2 and self.image_layer.ndim != 3:
            raise RuntimeError("Only 2D and 3D images are supported.")

    def set_sam_logits(self):
        if self.image_layer.ndim == 2:
            self.sam_logits = None
        else:
            self.sam_logits = [None] * self.image_layer.data.shape[0]

    #### auto
    def activate_annotation_mode_auto(self):
        self.sam_anything_predictor = SamAutomaticMaskGenerator(self.sam_model,
                                                                points_per_side=int(self.ui_elements.le_points_per_side.text()),
                                                                points_per_batch=int(self.ui_elements.le_points_per_batch.text()),
                                                                pred_iou_thresh=float(self.ui_elements.le_prediction_iou_threshold.text()),
                                                                stability_score_thresh=float(self.ui_elements.le_stability_score_threshold.text()),
                                                                stability_score_offset=float(self.ui_elements.le_stability_score_offset.text()),
                                                                box_nms_thresh=float(self.ui_elements.le_box_nms_threshold.text()),
                                                                crop_n_layers=int(self.ui_elements.le_crop_n_layers.text()),
                                                                crop_nms_thresh=float(self.ui_elements.le_crop_nms_threshold.text()),
                                                                crop_overlap_ratio=float(self.ui_elements.le_crop_overlap_ratio.text()),
                                                                crop_n_points_downscale_factor=int(self.ui_elements.le_crop_n_points_downscale_factor.text()),
                                                                min_mask_region_area=int(self.ui_elements.le_minimum_mask_region_area.text()),
                                                                )
        prediction = self.predict_everything()
        print(prediction)
        self.label_layer.data = prediction


    #### click
    def activate_annotation_mode_click(self):
            if self.image_layer.ndim == 2:
                self.point_size = max(int(np.min(self.image_layer.data.shape[:2]) / 100), 1)
            else:
                self.point_size = 2

            self.create_label_color_mapping()

            """ TODO: add again when done with history
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._history_limit = self.label_layer._history_limit
            self._reset_history()
             """

            self.image_layer.events.contrast_limits.connect(qdebounced(self.on_contrast_limits_change, timeout=1000))

            self.set_image()
            
            """ # TODO: add again
            self.update_points_layer(None)
            self.viewer.mouse_drag_callbacks.append(self.callback_click)
            self.viewer.keymap['Delete'] = self.on_delete
            self.label_layer.keymap['Control-Z'] = self.on_undo
            self.label_layer.keymap['Control-Shift-Z'] = self.on_redo
             """

    def create_label_color_mapping(self, num_labels=1000):
        self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
        for label in range(num_labels):
            color = self.label_layer.get_color(label)
            self.label_color_mapping["label_mapping"][label] = color
            self.label_color_mapping["color_mapping"][str(color)] = label

    def set_image(self):
        contrast_limits = self.image_layer.contrast_limits
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            self.sam_features = self.extract_feature_embeddings_2D(image, contrast_limits)
        
        elif self.image_layer.ndim == 3:
            total_slices = self.image_layer.data.shape[0]

            self.ui_elements.create_progress_bar(total_slices, "Creating SAM image embedding:")

            self.sam_features = []
            for index in range(total_slices):
                image_slice = np.asarray(self.image_layer.data[index, ...])

                self.sam_features.append(self.extract_feature_embeddings_2D(image_slice, contrast_limits))
                self.ui_elements.update_progress_bar(index+1)

            self.ui_elements.delete_progress_bar()


    def extract_feature_embeddings_2D(self, image, contrast_limits):
        image = self.prepare_image_for_sam(image, contrast_limits)
        self.sam_predictor.set_image(image)
        return self.sam_predictor.features

    def predict_everything_2D(self, image, contrast_limits):
        image = self.prepare_image_for_sam(image, contrast_limits)
        records = self.sam_anything_predictor.generate(image)#
        masks = np.asarray([record["segmentation"] for record in records])#
        print(masks.shape)
        prediction = np.argmax(masks, axis=0)
        return prediction

    def prepare_image_for_sam(self, image, contrast_limits):
        if not self.image_layer.rgb:
            image = np.stack((image,) * 3, axis=-1)  # Expand to 3-channel image
        image = image[..., :3]  # Remove a potential alpha channel
        image = normalize(image, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)#
        return image
    

    def predict_everything(self):
        contrast_limits = self.image_layer.contrast_limits
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            prediction = self.predict_everything_2D(image, contrast_limits)
        elif self.image_layer.ndim == 3:
            self.ui_elements.create_progress_bar(self.image_layer.data.shape[0], "Predicting everything:")
            prediction = []
            for index in tqdm(range(self.image_layer.data.shape[0]), desc="Predicting everything"):
                image_slice = np.asarray(self.image_layer.data[index, ...])
                prediction_slice = self.predict_everything_2D(image_slice, contrast_limits)
                prediction.append(prediction_slice)
                self.ui_elements.update_progress_bar(index+1)
                QApplication.processEvents()
            self.ui_elements.delete_progress_bar()
            prediction = np.asarray(prediction)
            prediction = self.merge_classes_over_slices(prediction)
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction
    
    def merge_classes_over_slices(self, prediction, threshold=0.5):  # Currently only computes overlap from next_slice to current_slice but not vice versa
        for i in range(prediction.shape[0] - 1):
            current_slice = prediction[i]
            next_slice = prediction[i+1]
            next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
            next_label_counts = next_label_counts[next_labels != 0]
            next_labels = next_labels[next_labels != 0]
            new_next_slice = np.zeros_like(next_slice)
            if len(next_labels) > 0:
                for next_label, next_label_count in zip(next_labels, next_label_counts):
                    current_roi_labels = current_slice[next_slice == next_label]
                    current_roi_labels, current_roi_label_counts = np.unique(current_roi_labels, return_counts=True)
                    current_roi_label_counts = current_roi_label_counts[current_roi_labels != 0]
                    current_roi_labels = current_roi_labels[current_roi_labels != 0]
                    if len(current_roi_labels) > 0:
                        current_max_count = np.max(current_roi_label_counts)
                        current_max_count_label = current_roi_labels[np.argmax(current_roi_label_counts)]
                        overlap = current_max_count / next_label_count
                        if overlap >= threshold:
                            new_next_slice[next_slice == next_label] = current_max_count_label
                        else:
                            new_next_slice[next_slice == next_label] = next_label
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                prediction[i+1] = new_next_slice
        return prediction
    
    """ chatgpt created # TODO: control correctness
    def merge_classes_over_slices(self, prediction, threshold=0.5):
        for i in range(prediction.shape[0] - 1):
            current_slice, next_slice = prediction[i], prediction[i+1]
            unique_next_labels = np.unique(next_slice)
            for label in unique_next_labels:
                if label != 0:  # ignore the background
                    overlapping_labels = current_slice[next_slice == label]
                    label_counter = Counter(overlapping_labels[overlapping_labels != 0])
                    if label_counter:  # if not empty
                        most_common_label, most_common_count = label_counter.most_common(1)[0]
                        overlap = most_common_count / np.sum(overlapping_labels == label)
                        if overlap >= threshold:
                            next_slice[next_slice == label] = most_common_label
        return prediction
     """

    def on_contrast_limits_change(self):
        self.set_image()

    def deactivate(self):
        pass