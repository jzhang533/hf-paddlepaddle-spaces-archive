import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="food_classification")

def inference(img):
  images = [cv2.imread(img)]
  results = classifier.predict(images=images)
  print(results)
  return results[0]

  
title="food_classification"
description="Food_classification, the model recognizes apple pie, short ribs, toast, beef patties, beef tartare. The PaddleHub Module supports API prediction and command line prediction."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)