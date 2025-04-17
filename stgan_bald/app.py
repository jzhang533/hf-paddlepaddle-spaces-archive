import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

stgan_bald = hub.Module(name="stgan_bald")


def inference(content):
  result = stgan_bald.bald(images=[cv2.imread(content)])
  print(result)
  return result[0]['data_0'],result[0]['data_1'],result[0]['data_2']

  
title="STGAN"
description="stgan_bald uses STGAN as a model and is trained using the CelebA dataset. The model can automatically generate 1-year, 3-year, and 5-year baldness effects based on images."

gr.Interface(inference,[gr.inputs.Image(type="filepath")],["image","image","image"],title=title,description=description).launch(enable_queue=True)