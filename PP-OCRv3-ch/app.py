import tempfile
import os

import gradio as gr
import paddlehub as hub
from PIL import Image

pp_ocrv3 = hub.Module(name="ch_pp-ocrv3")

def inference(img):
  with tempfile.TemporaryDirectory() as tempdir_name:
    pp_ocrv3.recognize_text(paths=[img],use_gpu=False,output_dir=tempdir_name,visualization=True)
    result_names = os.listdir(tempdir_name)
    output_image = Image.open(os.path.join(tempdir_name, result_names[0]))
    return [output_image]
  
title="ch_PP-OCRv3"
description="ch_PP-OCRv3 is a practical ultra-lightweight OCR system developed by PaddleOCR."

examples=[['test.png']]

gr.Interface(inference,gr.inputs.Image(type="filepath"),outputs=[gr.Gallery(label="Result", show_label=False).style(grid=[1, 1], height="auto")],title=title,description=description,examples=examples).launch(enable_queue=True)