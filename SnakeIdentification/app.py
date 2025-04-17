import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="SnakeIdentification")

def inference(img):
  images = [cv2.imread(img)]
  results = classifier.predict(images=images)
  print(results)
  return results[0]

  
title="SnakeIdentification"
description="SnakeIdentification, this model can accurately identify the species of snakes and accurately judge the toxicity of snakes. The PaddleHub Module supports API prediction and command line prediction."

examples=[['snake.jpeg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"text",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)