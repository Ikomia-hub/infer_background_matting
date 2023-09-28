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
import copy
import requests
import cv2
import torch
from torch.nn.modules.module import *
from torch.nn import functional as F
from pathlib import Path
from numpy import asarray
from infer_background_matting.model.model import MattingBase, MattingRefine
import numpy as np
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class InferBckMattingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_type = "mattingrefine"
        self.model_backbone = "mobilenetv2"
        self.model_backbone_scale = 0.25
        self.model_refine_mode = "sampling"
        self.model_refine_pixels = 80000
        self.model_refine_threshold = 0.7
        self.cuda = 'cuda'
        self.update = False

    def set_values(self, param_map):
        self.model_type = param_map["model_type"]
        self.model_backbone = param_map["model_backbone"]
        self.cuda = param_map["cuda"]
        self.model_backbone_scale = float(param_map["model_backbone_scale"])
        self.model_refine_threshold = float(param_map["model_refine_threshold"])
        self.model_refine_mode = param_map["model_refine_mode"]
        self.model_refine_pixels = int(param_map["model_refine_pixels"])
        self.update = utils.strtobool(param_map["update"])

    def get_values(self):
        param_map = {}
        param_map["model_type"] = self.model_type
        param_map["model_backbone"] = self.model_backbone
        param_map["cuda"] = self.cuda
        param_map["model_refine_mode"] = self.model_refine_mode
        param_map["model_backbone_scale"] = str(self.model_backbone_scale)
        param_map["model_refine_threshold"] = str(self.model_refine_threshold)
        param_map["model_refine_pixels"] = str(self.model_refine_pixels)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class InferBckMatting(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the process here
        input_img = dataprocess.CImageIO()
        input_img.description = "Img - " + input_img.description
        input_bck = dataprocess.CImageIO()
        input_bck.description = "Bck - " + input_bck.description
        input_optional_bck = dataprocess.CImageIO()
        input_optional_bck.description = "Bck to integrate on the image - " + input_optional_bck.description

        output_composite = dataprocess.CImageIO()
        output_alpha = dataprocess.CImageIO()
        output_fgr = dataprocess.CImageIO()
        output_err = dataprocess.CImageIO()
        output_composite.description = "Composite output - " + output_composite.description
        output_alpha.description = "Alpha output - " + output_alpha.description
        output_fgr.description = "Foreground output - " + output_fgr.description
        output_err.description = "Error output - " + output_err.description
        self.add_input(input_img)
        self.add_input(input_bck)
        self.add_input(input_optional_bck)
        self.add_output(output_composite)
        self.add_output(output_alpha)
        self.add_output(output_fgr)
        self.add_output(output_err)

        self.model = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferBckMattingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        param = self.get_param_object()
        # program run place
        self.device_ = param.cuda
        self.device = torch.device(self.device_)

        if self.device_ == 'cuda' and torch.cuda.is_available() == 0:
            self.device_ = 'cpu'
            self.device = torch.device(self.device_)
            print("cuda is not available on your machine, we pass in cpu")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 7

    # function to download model on google drive
    def download_file_from_google_drive(self, file_id, destination):
        path = os.path.join(os.path.dirname(__file__), "download_model")
        if not os.path.exists(path):
            os.makedirs(path)

        url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(url, params={'id': file_id, "confirm": "t"}, stream=True)
        self.save_response_content(response, destination)

    @staticmethod
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    # loading of model weights
    def model_treatment(self):
        param = self.get_param_object()
        if self.model is None or param.update:
            if param.model_type == 'mattingbase':
                self.model = MattingBase(backbone=param.model_backbone)
            elif param.model_type == 'mattingrefine':
                self.model = MattingRefine(
                    param.model_backbone,
                    param.model_backbone_scale,
                    param.model_refine_mode,
                    param.model_refine_pixels,
                    param.model_refine_threshold,
                    3)

            self.model.to(self.device).eval()

            if param.model_backbone == "resnet101":
                model_path = os.path.join(os.path.dirname(__file__), "download_model", "resnet101.pth")
                if not os.path.isfile(model_path):
                    self.download_file_from_google_drive("1zysR-jW6jydA2zkWfevxD1JpQHglKG1_", model_path)

                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

            elif param.model_backbone == "resnet50":
                model_path = os.path.join(os.path.dirname(__file__), "download_model", "resnet50.pth")
                if not os.path.isfile(model_path):
                    self.download_file_from_google_drive("1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf", model_path)

                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

            else:
                model_path = os.path.join(os.path.dirname(__file__), "download_model", "mobilenetv2.pth")
                if not os.path.isfile(model_path):
                    self.download_file_from_google_drive("1b2FQH0yULaiBwe4ORUvSxXpdWLipjLsI", model_path)

                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

            param.update = False

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        input_img = self.get_input(0)
        input_bck = self.get_input(1)
        input_bck_integration = self.get_input(2)
        img = input_img.get_image()
        bck = input_bck.get_image()
        h, w = np.shape(img)[:2]
        # height and width of input image must be divisible by 4
        if h % 4 != 0 or w % 4 != 0:
            h, w = h // 4 * 4, w // 4 * 4
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        if bck.shape != img.shape:
            bck = cv2.resize(bck, (w, h), interpolation=cv2.INTER_LINEAR)

        bck_integration = input_bck_integration.get_image()
        self.emit_step_progress()

        # resize of the optional bck
        if input_bck_integration.is_data_available():
            if img.shape != bck_integration.shape:
                dim = tuple(img[:, :, 0].shape)
                a, b = dim[0], dim[1]
                final = b, a
                bck_integration = cv2.resize(bck_integration, final, interpolation=cv2.INTER_LINEAR)

        # Get parameters
        param = self.get_param_object()
        # Get output
        output_composite = self.get_output(0)
        output_alpha = self.get_output(1)
        output_fgr = self.get_output(2)
        output_err = self.get_output(3)
        self.emit_step_progress()

        self.model_treatment()
        self.emit_step_progress()

        # conversion loop
        with torch.no_grad():
            # converting values from my arrays to float between 0 and 1 (model format)
            img_np = asarray([img])
            bck_np = asarray([bck])
            img_np = img_np.astype(np.float32)
            bck_np = bck_np.astype(np.float32)
            img_np = img_np / 255
            bck_np = bck_np / 255
            img_np = torch.from_numpy(img_np).permute(0, 3, 1, 2)
            bck_np = torch.from_numpy(bck_np).permute(0, 3, 1, 2)
            self.emit_step_progress()

            # with bck integration
            if input_bck_integration.is_data_available():
                bck_integration_np = asarray([bck_integration])
                bck_integration_np = bck_integration_np.astype(np.float32)
                bck_integration_np = bck_integration_np / 255
                bck_integration_tensor = torch.from_numpy(bck_integration_np).permute(0, 3, 1, 2)
                bck_integration_tensor = bck_integration_tensor.to(self.device, non_blocking=True)

            # passing our data into the model
            src = img_np.to(self.device, non_blocking=True)
            bgr = bck_np.to(self.device, non_blocking=True)
            self.emit_step_progress()

            if param.model_type == 'mattingbase':
                alpha, fgr, err, hid = self.model(src, bgr)
            elif param.model_type == 'mattingrefine':
                alpha, fgr, _, _, err, ref = self.model(src, bgr)

            composite = torch.cat([fgr * alpha.ne(0), alpha], dim=1)
            err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)

            # converting model outputs in ikomia output format
            output_composite_npy = (composite.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            output_err_npy = (err.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            output_fgr_npy = (fgr.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            output_alpha_npy = (alpha.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

            # delete a dimension
            output_fgr_npy = output_fgr_npy[0, :, :, :]
            output_err_npy = output_err_npy[0, :, :, :]
            output_composite_npy = output_composite_npy[0, :, :, :]
            output_alpha_npy = output_alpha_npy[0, :, :, :]
            self.emit_step_progress()

            # background integration
            if input_bck_integration.is_data_available():
                output_composite_f = fgr * alpha + bck_integration_tensor * (1 - alpha)
                output_composite_inter = (output_composite_f.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")[0,
                                         :, :, :]
                output_composite.set_image(output_composite_inter)
            else:
                output_composite.set_image(output_composite_npy)

            # outputs recovery
            output_err.set_image(output_err_npy)
            output_fgr.set_image(output_fgr_npy)
            output_alpha.set_image(output_alpha_npy)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class InferBckMattingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_background_matting"
        self.info.short_description = "Real-Time High-Resolution Background Matting"
        self.info.authors = "Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, " \
                            "Ira Kemelmacher-Shlizerman"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Background"
        self.info.version = "1.1.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.article = "Real-Time High-Resolution Background Matting"
        self.info.journal = "Computer Vision and Pattern Recognition"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.icon_path = "icon/image.png"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2012.07810"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_background_matting"
        self.info.original_repository = "https://github.com/PeterL1n/BackgroundMattingV2"
        # Keywords used for search
        self.info.keywords = "background,matting,refinement,alpha,foreground"

    def create(self, param=None):
        # Create process object
        return InferBckMatting(self.info.name, param)
