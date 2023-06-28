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

from napari_sam._ui_elements import UiElements
from napari_sam.slicer import slicer
from napari_sam.utils import normalize
from segment_anything import (
    SamPredictor,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
)
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    BBOX = 2
    AUTO = 3

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

    def load_model(self, model_type):
        if not torch.cuda.is_available():
            if not torch.backends.mps.is_available():
                self.device = "cpu"
            else:
                self.device = "mps"
        else:
            self.device = "cuda"
        sam_model = SAM_MODELS[model_type]["model"](self.get_weights_path(model_type))

        sam_model.to(self.device)
        print("debug 4")
        sam_predictor = SamPredictor(sam_model)
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

""" 
    def activate(self, annotator_mode):
        self.set_layers()
        self.adjust_image_layer_shape()
        self.check_image_dimension()
        self.set_sam_logits()

        if annotator_mode != AnnotatorMode.AUTO:
            self.activate_annotation_mode_click()

        elif annotator_mode == AnnotatorMode.AUTO:
            self.activate_annotator_mode_auto()

    # good
    def set_layers(self):
        self.image_name = self.ui_elements.cb_input_image_selctor.currentText()
        self.image_layer = self.viewer.layers[self.ui_elements.cb_input_image_selctor.currentText()]
        self.label_layer = self.viewer.layers[self.ui_elements.cb_output_label_selctor.currentText()]
        self.label_layer_changes = None
    # good
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
    # good
    def check_image_dimension(self):
        if self.image_layer.ndim != 2 and self.image_layer.ndim != 3:
            raise RuntimeError("Only 2D and 3D images are supported.")
    # good
    def set_sam_logits(self):
        if self.image_layer.ndim == 2:
            self.sam_logits = None
        else:
            self.sam_logits = [None] * self.image_layer.data.shape[0]
    # good

    def activate_annotation_mode_click(self):
            selected_layer = None
            if self.viewer.layers.selection.active != self.points_layer:
                selected_layer = self.viewer.layers.selection.active

            if selected_layer is not None:
                self.viewer.layers.selection.active = selected_layer

            if self.image_layer.ndim == 2:
                self.point_size = int(np.min(self.image_layer.data.shape[:2]) / 100)
                if self.point_size == 0:
                    self.point_size = 1
            else:
                self.point_size = 2

            self.create_label_color_mapping()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._history_limit = self.label_layer._history_limit
            self._reset_history()

            self.image_layer.events.contrast_limits.connect(qdebounced(self.on_contrast_limits_change, timeout=1000))

            self.set_image()
            self.update_points_layer(None)

            self.viewer.mouse_drag_callbacks.append(self.callback_click)
            self.viewer.keymap['Delete'] = self.on_delete
            self.label_layer.keymap['Control-Z'] = self.on_undo
            self.label_layer.keymap['Control-Shift-Z'] = self.on_redo

    def activate_annotator_mode_auto(self):
        self.sam_anything_predictor = SamAutomaticMaskGenerator(self.sam_model,
                                                                points_per_side=int(self.le_points_per_side.text()),
                                                                points_per_batch=int(self.le_points_per_batch.text()),
                                                                pred_iou_thresh=float(self.le_pred_iou_thresh.text()),
                                                                stability_score_thresh=float(self.le_stability_score_thresh.text()),
                                                                stability_score_offset=float(self.le_stability_score_offset.text()),
                                                                box_nms_thresh=float(self.le_box_nms_thresh.text()),
                                                                crop_n_layers=int(self.le_crop_n_layers.text()),
                                                                crop_nms_thresh=float(self.le_crop_nms_thresh.text()),
                                                                crop_overlap_ratio=float(self.le_crop_overlap_ratio.text()),
                                                                crop_n_points_downscale_factor=int(self.le_crop_n_points_downscale_factor.text()),
                                                                min_mask_region_area=int(self.le_min_mask_region_area.text()),
                                                                )
        prediction = self.predict_everything()
        self.label_layer.data = prediction

    def create_label_color_mapping(self):
        if self.label_layer is not None:
            self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
            for label in range(num_labels):
                color = self.label_layer.get_color(label)
                self.label_color_mapping["label_mapping"][label] = color
                self.label_color_mapping["color_mapping"][str(color)] = label

    def set_image(self):
        contrast_limits = self.image_layer.contrast_limits
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            self.sam_features = self.process_image(image, contrast_limits)
        
        elif self.image_layer.ndim == 3:
            total_slices = self.image_layer.data.shape[0]
            self.ui_elements.create_progress_bar(total_slices, "Creating SAM image embedding:")

            self.sam_features = []
            for index in range(total_slices):
                image_slice = np.asarray(self.image_layer.data[index, ...])
                self.sam_features.append(self.process_image(image_slice, contrast_limits))
                self.ui_elements.update_progress_bar(index+1)

            self.ui_elements.delete_progress_bar()
        
    def process_image(self, image_slice, contrast_limits):
        if not self.image_layer.rgb:
            image_slice = np.stack((image_slice,) * 3, axis=-1)  # Expand to 3-channel image
        image_slice = image_slice[..., :3]  # Remove a potential alpha channel
        image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
        self.sam_predictor.set_image(image_slice)
        return self.sam_predictor.features
 """