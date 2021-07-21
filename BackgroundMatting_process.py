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
import copy, torch, requests
from torch.nn.modules.module import *
from torch.nn import functional as F
from numpy import asarray
from model.model import MattingBase, MattingRefine


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class BackgroundMattingParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        self.model_type = "mattingbase"
        self.model_backbone = "mobilenetv2"
        self.model_backbone_scale = 0.25
        self.model_refine_mode = "full"
        self.model_refine_pixels = 80000
        self.model_refine_threshold = 0.7
        self.kernel_size = 3

    def setParamMap(self, paramMap):
        self.model_type = paramMap["model_type"]
        self.model_backbone = paramMap["model_backbone"]
        self.model_backbone_scale = int(paramMap["model_backbone"])
        self.model_refine_threshold = int(paramMap["model_refine_threshold"])
        self.model_refine_mode = paramMap["model_refine_mode"]
        self.model_refine_pixels = int(paramMap["model_refine_pixels"])
        self.kernel_size = int(paramMap["kernel_size"])

    def getParamMap(self):
        paramMap = core.ParamMap()
        paramMap["model_type"] = self.model_type
        paramMap["model_backbone"] = self.model_backbone
        paramMap["model_refine_mode"] = self.model_refine_mode
        paramMap["model_backbone_scale"] = str(self.model_backbone_scale)
        paramMap["model_refine_threshold"] = str(self.model_refine_threshold)
        paramMap["model_refine_pixels"] = str(self.model_refine_pixels)
        paramMap["kernel_size"] = str(self.kernel_size)
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class BackgroundMattingProcess(core.CProtocolTask):

    def __init__(self, name, param):
        core.CProtocolTask.__init__(self, name)
        # Add input/output of the process here
        self.addInput(dataprocess.CImageProcessIO())
        self.addInput(dataprocess.CImageProcessIO())
        self.addOutput(dataprocess.CImageProcessIO())
        self.addOutput(dataprocess.CImageProcessIO())
        self.addOutput(dataprocess.CImageProcessIO())
        self.addOutput(dataprocess.CImageProcessIO())

        # Create parameters class
        if param is None:
            self.setParam(BackgroundMattingParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        input_img = self.getInput(0)
        input_bck = self.getInput(1)
        img = input_img.getImage()
        bck = input_bck.getImage()
        # get param
        param = self.getParam()
        print("Start BackgroundMatting...")
        # Verification if the input is empty
        # if not input_img.isDataAvailable() or input_bck.isDataAvailable():
        #    raise ValueError("your input is empty, restart the task")
        # Get output
        output_composite = self.getOutput(0)
        output_alpha = self.getOutput(1)
        output_fgr = self.getOutput(2)
        output_err = self.getOutput(3)
        if torch.cuda.is_available():
            device_ = 'cuda'
        else:
            device_ = 'cpu'
        device = torch.device(device_)

        if param.model_type == 'mattingbase':
            model = MattingBase(param.model_backbone)
        if param.model_type == 'mattingrefine':
            model = MattingRefine(
                param.model_backbone,
                param.model_backbone_scale,
                param.model_refine_mode,
                param.model_refine_sample_pixels,
                param.model_refine_threshold,
                param.model_refine_kernel_size)
        '''#download models
        if param.model_backbone == "resnet101":
            if os.path.isfile(os.path.dirname(__file__)+"/download_model/resnet101.pth"):
                pass
            else:
                with open("C:\\Users\\Julien TEXIER\\Ikomia\\Plugins\\Python\\BackgroundMatting\\download_model\\resnet101.pth", 'w') as fp:
                    download_model = requests.get("https://drive.google.com/file/d/1zysR-jW6jydA2zkWfevxD1JpQHglKG1_/view?usp=sharing")
                    fp.write(download_model.content)
        elif param.model_backbone == "resnet50":
            download_model = requests.get("https://drive.google.com/file/d/1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf/view?usp=sharing")
        else:
            download_model = requests.get("https://drive.google.com/file/d/1b2FQH0yULaiBwe4ORUvSxXpdWLipjLsI/view?usp=sharing")

'''

        model = model.to(device).eval()
        model.load_state_dict(

            torch.load("C:\\Users\\Julien TEXIER\\Ikomia\\Plugins\\Python\\BackgroundMatting\\pytorch_mobilenetv2.pth",
                       map_location=device), strict=False)

        # conversion loop
        with torch.no_grad():
            img = asarray([img])
            bck = asarray([bck])
            img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
            bck = torch.from_numpy(bck).permute(0, 3, 1, 2).float()
            src = img.to(device, non_blocking=True)
            bgr = bck.to(device, non_blocking=True)
            if param.model_type == 'mattingbase':
                alpha, fgr, err, _ = model(src, bgr)
            elif param.model_type == 'mattingrefine':
                alpha, fgr, _, _, err, ref = model(src, bgr)

            composite = torch.cat([fgr * alpha.ne(0), alpha], dim=1)
            err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)

            output_composite_npy = composite.squeeze().cpu().numpy()
            output_err_npy = err.squeeze().cpu().numpy()
            output_fgr_npy = fgr.squeeze().cpu().numpy()
            output_alpha_npy = alpha.squeeze().cpu().numpy()

            output_composite.setImage(output_composite_npy)
            output_err.setImage(output_err_npy)
            output_fgr.setImage(output_fgr_npy)
            output_alpha.setImage(output_alpha_npy)

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
        self.info.description = "We introduce a real-time, high-resolution background replacement technique which " \
                                "operates at 30fps in 4K resolution, and 60fps for HD on a modern GPU. Our technique " \
                                "is based on background matting, where an additional frame of the background is" \
                                " captured and used in recovering the alpha matte and the foreground layer. " \
                                "The main challenge is to compute a high-quality alpha matte, preserving strand-level " \
                                "hair details, while processing high-resolution images in real-time. " \
                                "To achieve this goal, we employ two neural networks; a base network computes a " \
                                "low-resolution result which is refined by a second network operating at " \
                                "high-resolution " \
                                "on selective patches. We introduce two largescale video and image matting datasets: " \
                                "VideoMatte240K and PhotoMatte13K/85. Our approach yields higher quality results " \
                                "compared to the previous state-of-the-art in background matting, while simultaneously" \
                                "yielding a dramatic boost in both speed and resolution."
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
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/abs/2012.07810"
        # Code source repository
        self.info.repository = "https://github.com/PeterL1n/BackgroundMattingV2"
        # Keywords used for search
        self.info.keywords = "background,matting,refinement"

    def create(self, param=None):
        # Create process object
        return BackgroundMattingProcess(self.info.name, param)
