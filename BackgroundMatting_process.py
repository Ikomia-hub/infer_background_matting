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

from ikomia import core, dataprocess
import copy, torch, requests, cv2
from torch.nn.modules.module import *
from torch.nn import functional as F
from pathlib import Path
from numpy import asarray
from model.model import MattingBase, MattingRefine
import numpy as np
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class BackgroundMattingParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        self.model_type = "mattingrefine"
        self.model_backbone = "mobilenetv2"
        self.model_backbone_scale = 0.25
        self.model_refine_mode = "sampling"
        self.model_refine_pixels = 80000
        self.model_refine_threshold = 0.7
        self.kernel_size = 3
        self.cuda = 'cuda'
        self.update = False

    def setParamMap(self, paramMap):
        self.model_type = paramMap["model_type"]
        self.model_backbone = paramMap["model_backbone"]
        self.cuda = paramMap["cuda"]
        self.model_backbone_scale = int(paramMap["model_backbone"])
        self.model_refine_threshold = int(paramMap["model_refine_threshold"])
        self.model_refine_mode = paramMap["model_refine_mode"]
        self.model_refine_pixels = int(paramMap["model_refine_pixels"])
        self.kernel_size = int(paramMap["kernel_size"])
        self.update = int(paramMap["update"])

    def getParamMap(self):
        paramMap = core.ParamMap()
        paramMap["model_type"] = self.model_type
        paramMap["model_backbone"] = self.model_backbone
        paramMap["cuda"] = self.cuda
        paramMap["model_refine_mode"] = self.model_refine_mode
        paramMap["model_backbone_scale"] = str(self.model_backbone_scale)
        paramMap["model_refine_threshold"] = str(self.model_refine_threshold)
        paramMap["model_refine_pixels"] = str(self.model_refine_pixels)
        paramMap["kernel_size"] = str(self.kernel_size)
        paramMap["update"] = str(self.update)
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class BackgroundMattingProcess(core.CProtocolTask):

    def __init__(self, name, param):
        core.CProtocolTask.__init__(self, name)
        # Add input/output of the process here
        input_img = dataprocess.CImageProcessIO()
        input_img.description = "Img - " + input_img.description
        input_bck = dataprocess.CImageProcessIO()
        input_bck.description = "Bck - " + input_bck.description
        input_optional_bck = dataprocess.CImageProcessIO()
        input_optional_bck.description = "Bck to integrate on the image - " + input_optional_bck.description

        output_composite = dataprocess.CImageProcessIO()
        output_alpha = dataprocess.CImageProcessIO()
        output_fgr = dataprocess.CImageProcessIO()
        output_err = dataprocess.CImageProcessIO()
        output_composite.description = "Composite output - " + output_composite.description
        output_alpha.description = "Alpha output - " + output_alpha.description
        output_fgr.description = "Foreground output - " + output_fgr.description
        output_err.description = "Error output - " + output_err.description
        self.addInput(input_img)
        self.addInput(input_bck)
        self.addInput(input_optional_bck)
        self.addOutput(output_composite)
        self.addOutput(output_alpha)
        self.addOutput(output_fgr)
        self.addOutput(output_err)

        self.model = None
        # Create parameters class
        if param is None:
            self.setParam(BackgroundMattingParam())
        else:
            self.setParam(copy.deepcopy(param))

        param2 = self.getParam()
        print(param2.cuda)
    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    # function to download model on google drive
    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        input_img = self.getInput(0)
        input_bck = self.getInput(1)
        input_bck_integration = self.getInput(2)
        img = input_img.getImage()
        bck = input_bck.getImage()
        bck_integration = input_bck_integration.getImage()
        print(img.shape)
        # resize of the optional bck
        if input_bck_integration.isDataAvailable():
            if img.shape != bck_integration.shape:
                dim = tuple(img[:, :, 0].shape)
                a, b = dim[0], dim[1]
                final = b, a
                bck_integration = cv2.resize(bck_integration, final, interpolation=cv2.INTER_LINEAR)

        # get param
        param = self.getParam()
        print("Start BackgroundMatting...")

        # Get output
        output_composite = self.getOutput(0)
        output_alpha = self.getOutput(1)
        output_fgr = self.getOutput(2)
        output_err = self.getOutput(3)
        # program run place
        device_ = param.cuda
        device = torch.device(device_)

        if device_ == 'cuda' and torch.cuda.is_available() == 0:
            device_ = 'cpu'
            device = torch.device(device_)
            print("cuda is not available on your machine, we pass in cpu")

        if self.model is None or param.update is True:
            if param.model_type == 'mattingbase':
                self.model = MattingBase(backbone=param.model_backbone)
            if param.model_type == 'mattingrefine':
                self.model = MattingRefine(
                    param.model_backbone,
                    param.model_backbone_scale,
                    param.model_refine_mode,
                    param.model_refine_pixels,
                    param.model_refine_threshold,
                    3)
            self.model.to(device).eval()
            if param.model_backbone == "resnet101":
                if os.path.isfile(os.path.dirname(__file__) + "/download_model/resnet101.pth"):
                    pass
                else:
                    self.download_file_from_google_drive("1zysR-jW6jydA2zkWfevxD1JpQHglKG1_", Path(
                        os.path.dirname(__file__) + "/download_model/resnet101.pth"))

                self.model.load_state_dict(
                    torch.load(Path(os.path.dirname(__file__) + "/download_model/resnet101.pth"),
                               map_location=device), strict=False)

            elif param.model_backbone == "resnet50":
                if os.path.isfile(os.path.dirname(__file__) + "/download_model/resnet101.pth"):
                    pass
                else:
                    self.download_file_from_google_drive("1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf", Path(
                        os.path.dirname(__file__) + "/download_model/resnet50.pth"))

                self.model.load_state_dict(torch.load(Path(os.path.dirname(__file__) + "/download_model/resnet50.pth"),
                                                      map_location=device), strict=False)

            else:
                if os.path.isfile(os.path.dirname(__file__) + "/download_model/mobilenetv2.pth"):
                    pass
                else:
                    self.download_file_from_google_drive("1b2FQH0yULaiBwe4ORUvSxXpdWLipjLsI", Path(
                        os.path.dirname(__file__) + "/download_model/mobilenetv2.pth"))

                self.model.load_state_dict(
                    torch.load(Path(os.path.dirname(__file__) + "/download_model/mobilenetv2.pth"),
                               map_location=device), strict=False)
            param.update = False

        # conversion loop
        with torch.no_grad():
            # converting values from my arrays to float between 0 and 1 (model format)
            img_np = asarray([img])
            print(img_np.shape)
            bck_np = asarray([bck])
            bck_integration_np = asarray([bck_integration])
            img_np = img_np.astype(np.float32)
            bck_np = bck_np.astype(np.float32)
            bck_integration_np = bck_integration_np.astype(np.float32)
            img_np = img_np / 255
            bck_np = bck_np / 255
            bck_integration_np = bck_integration_np / 255
            print(img_np.shape)
            img_np = torch.from_numpy(img_np).permute(0, 3, 1, 2)
            bck_np = torch.from_numpy(bck_np).permute(0, 3, 1, 2)
            bck_integration_tensor = torch.from_numpy(bck_integration_np).permute(0, 3, 1, 2)

            # passing our data into the model
            src = img_np.to(device, non_blocking=True)
            bgr = bck_np.to(device, non_blocking=True)
            bck_integration_tensor = bck_integration_tensor.to(device, non_blocking=True)
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

            # background integration
            if input_bck_integration.isDataAvailable():
                output_composite_f = fgr * alpha + bck_integration_tensor * (1 - alpha)
                output_composite_inter = (output_composite_f.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")[0,
                                         :, :, :]
                output_composite.setImage(output_composite_inter)
            else:
                output_composite.setImage(output_composite_npy)

            # outputs recovery
            output_err.setImage(output_err_npy)
            output_fgr.setImage(output_fgr_npy)
            output_alpha.setImage(output_alpha_npy)

            print("End of the process...")

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class BackgroundMattingProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "BackgroundMatting"
        self.info.shortDescription = "Real-Time High-Resolution Background Matting"
        self.info.description = "This algorithm is a real-time, high-resolution background replacement technique which " \
                                "operates at 30fps in 4K resolution, and 60fps for HD on a modern GPU. The technique " \
                                "is based on background matting, where an additional frame of the background is" \
                                " captured and used in recovering the alpha matte and the foreground layer. " \
                                "The main challenge is to compute a high-quality alpha matte, preserving strand-level " \
                                "hair details, while processing high-resolution images in real-time. " \
                                "To achieve this goal,two neural networks are employed: a base network computes a " \
                                "low-resolution result which is refined by a second network operating at " \
                                "high-resolution.It is possible to replace the basic background with a new one by" \
                                " adding it to the algorithm input. "
        self.info.authors = "Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, " \
                            "Ira Kemelmacher-Shlizerman"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.article = "Real-Time High-Resolution Background Matting"
        self.info.journal = "publication journal"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.iconPath = "icon/image.png"
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/abs/2012.07810"
        # Code source repository
        self.info.repository = "https://github.com/PeterL1n/BackgroundMattingV2"
        # Keywords used for search
        self.info.keywords = "background,matting,refinement,alpha,foreground"

    def create(self, param=None):
        # Create process object
        return BackgroundMattingProcess(self.info.name, param)
