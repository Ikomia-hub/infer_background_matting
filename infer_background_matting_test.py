import logging
import cv2
import numpy as np
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer background matting =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    input_img_1 = t.get_input(1)
    input_img_1.set_image(img)
    input_img_2 = t.get_input(2)
    input_img_2.set_image(np.zeros_like(img))
    params = t.get_param_object()
    for model_type in ["mattingbase", "mattingrefine"]:
        for backbone in ["resnet101", "resnet50", "mobilenetv2"]:
            params["model_type"] = model_type
            params["model_backbone"] = backbone
            # without update = 1, model is not updated between two tests
            params["update"] = 1
            t.set_parameters(params)
            yield run_for_test(t)
