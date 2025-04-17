import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="resnext101_32x16d_wsl")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="resnext101_32x16d_wsl"
description="Because human-labeled datasets are approaching their functional limits in scale, Facebook's developers employed a unique transfer learning study that uses hashtags as labels to train on datasets containing billions of social media images , which has made a major breakthrough for large-scale training to weakly supervised learning (Weakly Supervised Learning). On the ImageNet image recognition benchmark, ResNeXt101_32x16d_wsl achieves a Top-1 accuracy of 84.24%. The structure of the PaddleHub Module is ResNeXt101_32x16d_wsl, the input image size is 224 x 224 x 3, and it supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)
