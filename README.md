<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_background_matting/main/icon/image.png" alt="Algorithm icon">
  <h1 align="center">infer_background_matting</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_background_matting">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_background_matting">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_background_matting/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_background_matting.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm is a real-time, high-resolution background replacement technique which operates at 30fps in 4K resolution, and 60fps for HD on a modern GPU. The technique is based on background matting, where an additional frame of the background is captured and used in recovering the alpha matte and the foreground layer. 

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Set input image with background to replace
wf.set_image_input(
    url="https://raw.githubusercontent.com/Ikomia-hub/infer_background_matting/main/sample_image/image1.png",
    index=0
)

# Set original background input
wf.set_image_input(
    url="https://raw.githubusercontent.com/Ikomia-hub/infer_background_matting/main/sample_image/image1_bck (1).png",
    index=1
)

# Set new background input
wf.set_image_input(
    url="https://raw.githubusercontent.com/Ikomia-hub/infer_background_matting/main/sample_image/image1_bck (2).png",
    index=2
)

# Add background matting algorithm
bck_matting = wf.add_task(name="infer_background_matting", auto_connect=True)

# Run the workflow
wf.run()

# Display result
display(bck_matting.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
# Add background matting algorithm
bck_matting = wf.add_task(name="infer_background_matting", auto_connect=True)

bck_matting.set_parameters({
    "model_type": "mattingrefine",
    "model_backbone": "mobilenetv2",
    "model_backbone_scale": "0.25",
    "model_refine_mode": "sampling",
    "model_refine_pixels": "80000",
    "model_refine_threshold": "0.7",
    "cuda": "cuda",
})

# Run the workflow
wf.run()
```

- **model_type** (str): choose either *"mattingbase"* or *"mattingrefine"* (default - higher quality)
- **model_backbone** (str): model backbone, can be *"mobilenetv2"* (default), *"resnet50"* or *"resnet101"*
- **model_backbone_scale** (float): image downsample scale for passing through backbone (default 0.25)
- **model_refine_mode** (str): refine area selection mode
    - *"full"*: no area selection, refine everywhere using regular Conv2d
    - *"sampling"*: refine fixed amount of pixels ranked by the top most errors (default)
    - *"thresholding"*: refine varying amount of pixels that has more error than the threshold
- **model_refine_pixels** (int): only used when *model_refine_mode = "sampling"* (default 80000)
- **model_refine_threshold** (float [0 - 1]): only used when *model_refine_mode = "thresholding"* (default 0.7)
- **cuda** (str): "cuda" (default) to execute with CUDA acceleration or "cpu"

***Note***: parameter key and value should be in **string format** when added to the dictionary.


## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
# Add background matting algorithm
bck_matting = wf.add_task(name="infer_background_matting", auto_connect=True)

# Run the workflow
wf.run()

# Iterate over outputs
for output in bck_matting.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

Background matting algorithm generates 4 outpus:

1. Composite image (CImageIO)
2. Alpha (CImageIO)
3. Foreground image (CImageIO)
4. Error (CImageIO)