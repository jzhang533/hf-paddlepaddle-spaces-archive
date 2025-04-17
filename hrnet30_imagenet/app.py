import gradio as gr
import paddlehub as hub


model = hub.Module(name='hrnet30_imagenet')
def inference(img):
  result = model.predict([img])
  for key, value in result[0].items():
    result[0][key] = float(value)
  print(result)
  return result[0]

  
title="hrnet30_imagenet"
description="HRNet is a new neural network proposed by Microsoft Research Asia in 2019. Unlike previous convolutional neural networks, this network can still maintain high resolution in the deep layers of the network, so the heatmap of predicted keypoints is more accurate, and it is also more spatially accurate. Furthermore, the network performs particularly well in other resolution-sensitive vision tasks, such as detection and segmentation."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)