import logging
import cv2
import numpy as np
from ikomia.core import task

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer background matting =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img_0 = t.getInput(0)
    input_img_0.setImage(img)
    input_img_1 = t.getInput(1)
    input_img_1.setImage(img)
    input_img_2 = t.getInput(2)
    input_img_2.setImage(np.zeros_like(img))
    params = task.get_parameters(t)
    for model_type in ["mattingbase", "mattingrefine"]:
        for backbone in ["resnet101", "resnet50", "mobilenetv2"]:
            params["model_type"] = model_type
            params["model_backbone"] = backbone
            # without update = 1, model is not updated between two tests
            params["update"] = 1
            task.set_parameters(t, params)
            t.run()