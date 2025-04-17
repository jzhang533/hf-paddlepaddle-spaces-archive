import gradio as gr
import paddlehub as hub


model = hub.Module(name='ghostnet_x1_3_imagenet')
    
def inference(img):
  result = model.predict([img])
  for key, value in result[0].items():
    result[0][key] = float(value)
  print(result)
  return result[0]

  
title="ghostnet_x1_3_imagenet"
description="GhostNet is a new lightweight network structure proposed by Huawei in 2020. By introducing the ghost module, the redundant calculation problem of features in traditional deep networks is greatly alleviated, and the network parameters and calculation amount are greatly reduced."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)