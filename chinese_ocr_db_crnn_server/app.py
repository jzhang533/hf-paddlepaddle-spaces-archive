import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image

ocr = hub.Module(name="chinese_ocr_db_crnn_server")

def inference(img):
  result = ocr.recognize_text(images=[cv2.imread(img)],use_gpu=False,output_dir='/home/user/app/',visualization=True)
  return result[0]['save_path']
  
title="chinese_ocr_db_crnn_server"
description="chinese_ocr_db_crnn_server Module is used to identify Chinese characters in pictures"

examples=[['test.png']]

gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="file"),title=title,description=description,examples=examples).launch(enable_queue=True,debug=True)