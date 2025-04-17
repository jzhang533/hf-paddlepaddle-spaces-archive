import gradio as gr
import cv2
import paddlehub as hub


model = hub.Module(name='U2Net')
    
    
def inference(img):
  result = model.Segmentation(
      images=[cv2.imread(img)],
      paths=None,
      batch_size=1,
      input_size=320,
      output_dir='output',
      visualization=True)
  print(result)
  return result[0]['front'][:,:,::-1], result[0]['mask']

outputs = [
           gr.outputs.Image(type="numpy",label="Front"),
           gr.outputs.Image(type="numpy",label="Mask")
           ]
title="u2Net"
description="U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath",shape=(512,512)),outputs,title=title,description=description,examples=examples).launch(enable_queue=True)