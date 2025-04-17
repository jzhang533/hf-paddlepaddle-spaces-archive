import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

face_detector = hub.Module(name="pyramidbox_lite_mobile")

def inference(img):
  face_detector.face_detection(images=[cv2.imread(img)],use_gpu=False,visualization=True)
  return './output/0.png'

  
title="pyramidbox_lite_mobile"
description="PyramidBox-Lite is a lightweight model developed based on the paper PyramidBox published by Baidu at the top computer vision conference ECCV 2018 in 2018. The model is based on the backbone network FaceBoxes, which is very strong for common problems such as illumination, mask occlusion, expression changes, and scale changes. robustness. The PaddleHub Module is an optimized model for mobile terminals, suitable for deployment on devices with limited computing power such as mobile terminals or edge detection. It is trained based on the WIDER FACE dataset and Baidu self-collected face dataset to support prediction. Can be used for face detection."

examples=[['groot.png']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="file"),title=title,description=description,examples=examples).launch(enable_queue=True)