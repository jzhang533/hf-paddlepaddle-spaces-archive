import gradio as gr
import paddlehub as hub
import cv2

model = hub.Module(name="U2Net_Portrait")


def inference(img):
  result = model.Portrait_GEN(images=[cv2.imread(img)])
  print(result)
  return result[0][:,:,::-1]

  
title="U2Net_Portrait"
description="U2Net_Portrait can be used to extract face sketch results."

examples=[['groot.png']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True)
