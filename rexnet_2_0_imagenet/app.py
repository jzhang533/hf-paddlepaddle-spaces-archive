import gradio as gr
import paddlehub as hub

model = hub.Module(name='rexnet_2_0_imagenet')

def inference(img):
  result = model.predict([img])
  print(result)
  for key, value in result[0].items():
    result[0][key] = float(value)
  return result[0]

  
title="rexnet_2_0_imagenet"
description="ReXNet is a network designed based on new network design principles proposed by NAVER AI Lab. The authors propose a set of design principles for the problem of representative bottlenecks in existing networks, and they argue that conventional designs create representative bottlenecks, which can affect model performance. To investigate the representation bottleneck, the authors investigate the matrix rank of features generated by ten thousand random networks. In addition, the channel configuration of the whole layer is also studied to design a more accurate network architecture. Finally, the authors propose a set of simple and effective design principles to alleviate representation bottlenecks."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)