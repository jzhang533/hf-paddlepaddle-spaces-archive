import gradio as gr
import paddlehub as hub


model = hub.Module(name='repvgg_b2g4_imagenet')


def inference(img):
  result = model.predict([img])
  for key, value in result[0].items():
    result[0][key] = float(value)
  print(result)
  return result[0]

  
title="repvgg_b2g4_imagenet"
description="The RepVGG (Making VGG-style ConvNets Great Again) series model is a simple but functional model proposed by Tsinghua University (Ding Guiguang's team), Megvii Technology (Sun Jian, etc.), Hong Kong University of Science and Technology and Aberystwyth University in 2021. Powerful convolutional neural network architecture. There is an inference time agent similar to VGG. The body consists of 3x3 convolutions and relu stacks, while the training-time model has a multi-branch topology. The decoupling of training time and inference time is achieved through a reparameterization technique, so the model is called repvgg."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)