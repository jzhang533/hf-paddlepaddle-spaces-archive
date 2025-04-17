import gradio as gr
import paddlehub as hub
import cv2

model = hub.Module(name='UGATIT_92w', use_gpu=False)


def inference(img):
  result = model.style_transfer(images=[cv2.imread(img)])
  return result[0][:,:,::-1]

  
title="UGATIT_92w"
description="UGATIT image style conversion model, the model can convert the input face image into anime style."

examples=[['groot.png']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)