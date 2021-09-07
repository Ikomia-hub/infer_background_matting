# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from BackgroundMatting.BackgroundMatting_process import BackgroundMattingParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class BackgroundMattingWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = BackgroundMattingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)
        # Creation widget model_type
        self.label_model = QLabel("Model_type:")
        self.model_type = QComboBox()
        self.model_type.addItem("mattingbase")
        self.model_type.addItem("mattingrefine")
        self.grid_layout.addWidget(self.label_model, 1, 0)
        self.grid_layout.addWidget(self.model_type, 1, 1)
        self.model_type.setCurrentText(self.parameters.model_type)

        # Creation widget model_backbone
        self.label_model_back = QLabel("model_backbone:")
        self.model_backbone = QComboBox()
        self.model_backbone.addItem("resnet101")
        self.model_backbone.addItem("resnet50")
        self.model_backbone.addItem("mobilenetv2")
        self.grid_layout.addWidget(self.label_model_back, 2, 0)
        self.grid_layout.addWidget(self.model_backbone, 2, 1)
        self.model_backbone.setCurrentText(self.parameters.model_backbone)

        # Creation widget model_backbone_scale
        self.label_model_back_scale = QLabel("model_backbone_scale:")
        self.model_backbone_scale = QDoubleSpinBox()
        self.model_backbone_scale.setMinimum(0.0)
        self.model_backbone_scale.setMaximum(0.5)
        self.model_backbone_scale.setStepType(1)
        self.grid_layout.addWidget(self.label_model_back_scale, 3, 0)
        self.grid_layout.addWidget(self.model_backbone_scale, 3, 1)
        self.model_backbone_scale.setValue(self.parameters.model_backbone_scale)

        # Creation widget refine_mode
        self.label_model_refine_mode = QLabel("model_refine_mode:")
        self.model_refine_mode = QComboBox()
        self.model_refine_mode.addItem("full")
        self.model_refine_mode.addItem("sampling")
        self.model_refine_mode.addItem("thresholding")
        self.grid_layout.addWidget(self.label_model_refine_mode, 4, 0)
        self.grid_layout.addWidget(self.model_refine_mode, 4, 1)
        self.model_refine_mode.setCurrentText(self.parameters.model_refine_mode)
        self.model_refine_mode.currentTextChanged.connect(self.on_method_change)

        # Creation widget model_refine_threshold
        self.label_model_refine_threshold = QLabel("model_refine_threshold:")
        self.model_refine_threshold = QDoubleSpinBox()
        self.model_refine_threshold.setMinimum(0.0)
        self.model_refine_threshold.setMaximum(1.0)
        self.model_refine_threshold.setStepType(1)
        self.grid_layout.addWidget(self.label_model_refine_threshold, 5, 0)
        self.grid_layout.addWidget(self.model_refine_threshold, 5, 1)
        self.model_refine_threshold.setValue(self.parameters.model_refine_threshold)

        # Creation widget model_refine_pixels
        self.label_model_refine_pixels = QLabel("model_refine_pixels:")
        self.model_refine_pixels = QSpinBox()
        self.model_refine_pixels.setMinimum(50000)
        self.model_refine_pixels.setMaximum(100000)
        self.grid_layout.addWidget(self.label_model_refine_pixels, 5, 0)
        self.grid_layout.addWidget(self.model_refine_pixels, 5, 1)
        self.model_refine_pixels.setValue(self.parameters.model_refine_pixels)

        # Creation widget use_cuda
        self.label_model_Cuda = QLabel("Application on:")
        self.cuda = QComboBox()
        self.cuda.addItem("cuda")
        self.cuda.addItem("cpu")
        self.grid_layout.addWidget(self.label_model_Cuda, 6, 0)
        self.grid_layout.addWidget(self.cuda, 6, 1)
        self.cuda.setCurrentText(self.parameters.cuda)

    def on_method_change(self):
        if self.model_refine_mode.currentText() == "sampling":
            self.label_model_refine_pixels.show()
            self.model_refine_pixels.show()
            self.label_model_refine_threshold.hide()
            self.model_refine_threshold.hide()
        if self.model_refine_mode.currentText() == "thresholding":
            self.label_model_refine_pixels.hide()
            self.model_refine_pixels.hide()
            self.label_model_refine_threshold.show()
            self.model_refine_threshold.show()
        else:
            self.label_model_refine_pixels.hide()
            self.model_refine_pixels.hide()
            self.label_model_refine_threshold.hide()
            self.model_refine_threshold.hide()

    def onApply(self):
        self.parameters.model_type = self.model_type.currentText()
        self.parameters.model_backbone = self.model_backbone.currentText()
        self.parameters.model_refine_mode = self.model_refine_mode.currentText()
        self.parameters.cuda = self.cuda.currentText()
        self.parameters.model_backbone_scale = self.model_backbone_scale.value()
        self.parameters.model_refine_threshold = self.model_refine_threshold.value()
        self.parameters.model_refine_pixels = self.model_refine_pixels.value()
        self.parameters.update = True
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class BackgroundMattingWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "BackgroundMatting"

    def create(self, param):
        # Create widget object
        return BackgroundMattingWidget(param, None)
