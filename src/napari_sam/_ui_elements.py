from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRadioButton, QGroupBox, QProgressBar, QApplication, QScrollArea, QLineEdit, QCheckBox, QListWidget
from qtpy.QtGui import QIntValidator, QDoubleValidator
# from napari_sam.QCollapsibleBox import QCollapsibleBox
from qtpy import QtCore
from qtpy.QtCore import Qt
import napari
import numpy as np
from enum import Enum
from collections import deque, defaultdict
import inspect
from segment_anything import SamPredictor, build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from napari_sam.utils import normalize
import torch
from vispy.util.keys import CONTROL
import copy
import warnings
from tqdm import tqdm
from superqt.utils import qdebounced
from napari_sam.slicer import slicer
import urllib.request
from pathlib import Path
import os
from os.path import join



class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    AUTO = 2

SAM_MODELS = {
    "default": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model": build_sam_vit_l},
    "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model": build_sam_vit_b},
    "MedSAM": {"filename": "sam_vit_b_01ec64_medsam.pth", "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth", "model": build_sam_vit_b},
}

class LayerSelector(QListWidget):
    def __init__(self, viewer, layer_type):
        super().__init__()
        self.viewer = viewer
        self.layer_type = layer_type  # Save layer type
        self.viewer.events.layers_change.connect(self.update_layers)
        self.update_layers()  # Update layers at startup

        # connect a method to the itemSelectionChanged signal
        self.itemSelectionChanged.connect(self.item_selected)

    def update_layers(self, event=None):
        # Remove all items
        self.clear()

        # Add all label layers
        for layer in self.viewer.layers:
            if isinstance(layer, self.layer_type):  # Use the stored layer type
                self.addItem(layer.name)

    def item_selected(self):
        # get the text of the currently selected item
        selected_layer = self.currentItem().text()
        print(f'Selected label: {selected_layer}')

        
class UiElements:
    def __init__(self, viewer):
        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}
        self.viewer = viewer

        self.cached_models = None
        self.loaded_model = None

        self.main_layout = QVBoxLayout()
        self._init_main_layout()

        self.annotator_mode = AnnotatorMode.NONE


         # dict of model_type: bool(cached)
         # list of strings: model_types (cached/loaded)
        
        # TODO: figure out what is needed and remove the rest
        """
            self.image_name = None
            self.image_layer = None
            self.label_layer = None
            self.label_layer_changes = None
            self.label_color_mapping = None
            self.points_layer = None
            self.points_layer_name = "Ignore this layer1"  # "Ignore this layer <hidden>"
            self.old_points = np.zeros(0)

            self.sam_model = None
            self.sam_predictor = None
            self.sam_logits = None
            self.sam_features = None

            self.points = defaultdict(list)
            self.point_label = None
        """

    
    def _init_main_layout(self):
        #### adding UI elements
        #### model selection
        l_model_type = QLabel("Select model type:")
        self.main_layout.addWidget(l_model_type)

        self.cb_model_type = QComboBox() #### Callback function is defined below
        self.main_layout.addWidget(self.cb_model_type)
        
        
        self.btn_load_model = QPushButton("Load model") #### TODO: Callback function is defined through a setter function below 
        self.main_layout.addWidget(self.btn_load_model)

        self._update_model_selection_combobox_and_button()

        #### input image selection
        l_image_layer = QLabel("Select input image:")
        self.main_layout.addWidget(l_image_layer)

        self.label_selector = LayerSelector(self.viewer, napari.layers.Image) #### no callback function
        self.main_layout.addWidget(self.label_selector)

        #### label layer selection
        l_label_layer = QLabel("Select class:")
        self.main_layout.addWidget(l_label_layer)

        self.label_selector = LayerSelector(self.viewer, napari.layers.Labels) #### Callback function is defined through a setter function below
        self.main_layout.addWidget(self.label_selector)

        #### connecting signals for pure UI elements interactions
        self.cb_model_type.currentTextChanged.connect(self._update_model_selection_combobox_and_button)

        """ 
        #### input image selection



        self.cb_image_layers.currentTextChanged.connect(self.on_image_change)
        self.main_layout.addWidget(self.cb_image_layers)

        l_label_layer = QLabel("Select output labels layer:")
        self.main_layout.addWidget(l_label_layer)

        self.cb_label_layers = QComboBox()
        self.cb_label_layers.addItems(self.get_layer_names("labels"))
        self.main_layout.addWidget(self.cb_label_layers)

        self.comboboxes = [{"combobox": self.cb_image_layers, "layer_type": "image"}, {"combobox": self.cb_label_layers, "layer_type": "labels"}]

        self.g_annotation = QGroupBox("Annotation mode")
        self.l_annotation = QVBoxLayout()

        self.rb_click = QRadioButton("Click && Bounding Box")
        self.rb_click.setChecked(True)
        self.rb_click.setToolTip("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button \n \n"
                                 "Undo: Control + Z \n \n"
                                 "Select Point: Left Click \n \n"
                                 "Delete Selected Point: Delete")
        self.l_annotation.addWidget(self.rb_click)
        self.rb_click.clicked.connect(self.on_everything_mode_checked)

        self.rb_auto = QRadioButton("Everything")

        self.rb_auto.setToolTip("Creates automatically an instance segmentation \n"
                                            "of the entire image.\n"
                                            "No user interaction possible.")
        self.l_annotation.addWidget(self.rb_auto)
        self.rb_auto.clicked.connect(self.on_everything_mode_checked)

        self.g_annotation.setLayout(self.l_annotation)
        self.main_layout.addWidget(self.g_annotation)

        self.g_segmentation = QGroupBox("Segmentation mode")
        self.l_segmentation = QVBoxLayout()

        self.rb_semantic = QRadioButton("Semantic")
        self.rb_semantic.setChecked(True)
        self.rb_semantic.setToolTip("Enables the user to create a \n"
                                 "multi-label (semantic) segmentation of different classes.\n \n"
                                 "All objects from the same class \n"
                                 "should be given the same label by the user.\n \n"
                                 "The current label can be changed by the user \n"
                                 "on the labels layer pane after selecting the labels layer.")
        self.l_segmentation.addWidget(self.rb_semantic)

        self.rb_instance = QRadioButton("Instance")
        self.rb_instance.setToolTip("Enables the user to create an \n"
                                 "instance segmentation of different objects.\n \n"
                                 "Objects can be from the same or different classes,\n"
                                 "but each object should be given a unique label by the user. \n \n"
                                 "The current label can be changed by the user \n"
                                 "on the labels layer pane after selecting the labels layer.")
        self.l_segmentation.addWidget(self.rb_instance)

        self.rb_semantic.clicked.connect(self.on_segmentation_mode_changed)
        self.rb_instance.clicked.connect(self.on_segmentation_mode_changed)

        self.g_segmentation.setLayout(self.l_segmentation)
        self.main_layout.addWidget(self.g_segmentation)

        self.btn_activate = QPushButton("Activate")
        self.btn_activate.clicked.connect(self._activate)
        self.btn_activate.setEnabled(False)
        self.is_active = False
        self.main_layout.addWidget(self.btn_activate)

        self.btn_mode_switch = QPushButton("Switch to BBox Mode")
        self.btn_mode_switch.clicked.connect(self._switch_mode)
        self.btn_mode_switch.setEnabled(False)
        self.main_layout.addWidget(self.btn_mode_switch)

        self.check_prev_mask = QCheckBox('Use previous SAM prediction (recommended)')
        self.check_prev_mask.setEnabled(False)
        self.check_prev_mask.setChecked(True)
        self.main_layout.addWidget(self.check_prev_mask)

        self.check_auto_inc_bbox= QCheckBox('Auto increment bounding box label')
        self.check_auto_inc_bbox.setEnabled(False)
        self.check_auto_inc_bbox.setChecked(True)
        self.main_layout.addWidget(self.check_auto_inc_bbox)

        container_widget_info = QWidget()
        container_layout_info = QVBoxLayout(container_widget_info)

        self.g_size = QGroupBox("Point && Bounding Box Settings")
        self.l_size = QVBoxLayout()

        l_point_size = QLabel("Point Size:")
        self.l_size.addWidget(l_point_size)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_point_size = QLineEdit()
        self.le_point_size.setText("1")
        self.le_point_size.setValidator(validator)
        self.l_size.addWidget(self.le_point_size)

        l_bbox_edge_width = QLabel("Bounding Box Edge Width:")
        self.l_size.addWidget(l_bbox_edge_width)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_bbox_edge_width = QLineEdit()
        self.le_bbox_edge_width.setText("1")
        self.le_bbox_edge_width.setValidator(validator)
        self.l_size.addWidget(self.le_bbox_edge_width)
        self.g_size.setLayout(self.l_size)
        container_layout_info.addWidget(self.g_size)

        self.g_info_tooltip = QGroupBox("Tooltip Information")
        self.l_info_tooltip = QVBoxLayout()
        self.label_info_tooltip = QLabel("Every mode shows further information when hovered over.")
        self.label_info_tooltip.setWordWrap(True)
        self.l_info_tooltip.addWidget(self.label_info_tooltip)
        self.g_info_tooltip.setLayout(self.l_info_tooltip)
        container_layout_info.addWidget(self.g_info_tooltip)

        self.g_info_contrast = QGroupBox("Contrast Limits")
        self.l_info_contrast = QVBoxLayout()
        self.label_info_contrast = QLabel("SAM computes its image embedding based on the current image contrast.\n"
                                          "Image contrast can be adjusted with the contrast slider of the image layer.")
        self.label_info_contrast.setWordWrap(True)
        self.l_info_contrast.addWidget(self.label_info_contrast)
        self.g_info_contrast.setLayout(self.l_info_contrast)
        container_layout_info.addWidget(self.g_info_contrast)

        self.g_info_click = QGroupBox("Click Mode")
        self.l_info_click = QVBoxLayout()
        self.label_info_click = QLabel("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button\n \n"
                                 "Undo: Control + Z\n \n"
                                 "Select Point: Left Click\n \n"
                                 "Delete Selected Point: Delete\n \n"
                                 "Pick Label: Control + Left Click\n \n"
                                 "Increment Label: M\n \n")
        self.label_info_click.setWordWrap(True)
        self.l_info_click.addWidget(self.label_info_click)
        self.g_info_click.setLayout(self.l_info_click)
        container_layout_info.addWidget(self.g_info_click)

        scroll_area_info = QScrollArea()
        scroll_area_info.setWidget(container_widget_info)
        scroll_area_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_layout.addWidget(scroll_area_info)

        self.scroll_area_auto = self.init_auto_mode_settings()
        self.main_layout.addWidget(self.scroll_area_auto)

        
        
        self.point_size = 10
        self.le_point_size.setText(str(self.point_size))
        self.bbox_edge_width = 10
        self.le_bbox_edge_width.setText(str(self.bbox_edge_width))
         """
        
    def _set_model_load_handler(self, handler: callable) -> None:
        """
        Sets the callback function for the load model button.
        function signature: partial(handler, args)
        """
        self.btn_load_model.clicked.connect(handler)

    def _update_model_selection_combobox_and_button(self):
        """Updates the model selection combobox and button based on the cached models."""

        # Disconnect the signal if it's connected to avoid triggering the signal when clearing the combobox
        if self.cb_model_type.receivers(self.cb_model_type.currentTextChanged) > 0: 
            self.cb_model_type.currentTextChanged.disconnect()

        self.cached_models, combobox_models = get_cached_models(SAM_MODELS, self.loaded_model)

        # Store current selection
        current_selection = self.cb_model_type.currentText()

        # Check if the combobox needs to be updated
        if set(combobox_models) != set([self.cb_model_type.itemText(i) for i in range(self.cb_model_type.count())]):
            self.cb_model_type.clear()
            self.cb_model_type.addItems(combobox_models)

        # Reselect the previously selected item
        index = self.cb_model_type.findText(current_selection)
        self.cb_model_type.setCurrentIndex(index)

        if self.cached_models[list(self.cached_models.keys())[self.cb_model_type.currentIndex()]]:
            self.btn_load_model.setText("Load model")
        else:
            self.btn_load_model.setText("Download and load model")

        # Reconnect the signal
        self.cb_model_type.currentTextChanged.connect(self._update_model_selection_combobox_and_button)


def get_cached_models(SAM_MODELS: dict, loaded_model: str) -> tuple:
    """Check if the weights of the SAM models are cached locally."""
    model_types = list(SAM_MODELS.keys())
    cached_models: dict = {}
    cache_dir: str = str(Path.home() / ".cache/napari-segment-anything")

    for model_type in model_types:
        filename = os.path.basename(SAM_MODELS[model_type]["filename"])
        if os.path.isfile(os.path.join(cache_dir, filename)):
            cached_models[model_type] = True
        else:
            cached_models[model_type] = False

    """ creates a list of strings for the combobox """
    entries: list = []
    for name, is_cached in cached_models.items():
        if name == loaded_model:
            entries.append(f"{name} (Loaded)")
        elif is_cached:
            entries.append(f"{name} (Cached)")
        else:
            entries.append(f"{name} (Auto-Download)")
    return cached_models, entries


def update_image_layers(viewer):
    """
    This function updates the list of image layers in the viewer and returns those as a list.

    Parameters:
    viewer (napari.Viewer): The image viewer.

    Returns:
    list[napari.layers.Image]: The list of image layers in the viewer.
    """
    # Here I'm just getting the image layers as they are.
    # You may want to add, remove or modify layers based on your requirements.
    image_layers = [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]
    
    # Return the list of image layers
    return image_layers
