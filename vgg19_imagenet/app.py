import paddlehub as hub
import gradio as gr

module = hub.Module(name="vgg19_imagenet")

def inference(img):
  test_img_path = img
  
  # set input dict
  input_dict = {"image": [test_img_path]}
  
  # execute predict and print the result
  results = module.classification(data=input_dict)
  return results[0][0]
  
title="vgg19_imagenet"
description="VGG is an image classification model proposed by the Oxford University Computer Vision Group and DeepMind in 2014. This series of models explores the relationship between the depth of the convolutional neural network and its performance. It is experimentally proved that increasing the depth of the network can affect the final performance of the network to a certain extent. So far, VGG is still used by many other image tasks. BackBone network for feature extraction. The PaddleHub Module has a VGG19 structure and is trained on the ImageNet-2012 dataset."

gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description).launch(enable_queue=True)