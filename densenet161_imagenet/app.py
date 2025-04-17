import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="densenet161_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="densenet161_imagenet"
description="DenseNet is the model of the best paper in CVPR 2017. DenseNet connects each layer with other layers in a feed-forward manner, so that the L-layer network has L(L+1)/2 direct connections. For each layer, its input is the feature map of all previous layers, and its own feature map is used as the input of all subsequent layers. DenseNet alleviates the vanishing gradient problem, strengthens feature propagation, promotes feature reuse, and greatly reduces the amount of parameters. The PaddleHub Module structure is DenseNet161, based on ImageNet-2012 dataset training, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)