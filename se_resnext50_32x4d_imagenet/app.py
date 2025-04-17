import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="se_resnext50_32x4d_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="se_resnext50_32x4d_imagenet"
description="Squeeze-and-Excitation Networks is an image classification structure proposed by Momenta in 2017. This structure improves the accuracy by modeling the correlation between feature channels and strengthening important features. SE_ResNeXt adds SE Block based on the ResNeXt model and won the 2017 ILSVR competition. The structure of the PaddleHub Module is SE_ResNeXt50_32x4d. It is trained based on the ImageNet-2012 dataset. The input image size is 224 x 224 x 3. It supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)