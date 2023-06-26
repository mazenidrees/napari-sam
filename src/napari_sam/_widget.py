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