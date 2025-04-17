import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image

ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

def inference(img):
  result = ocr.recognize_text(images=[cv2.imread(img)],use_gpu=False,output_dir='/home/user/app/',visualization=True)
  return result[0]['save_path']
  
title="chinese_ocr_db_crnn_mobile"
description="chinese_ocr_db_crnn_mobile Module is used to identify Chinese characters in pictures. It continues to recognize the Chinese text in the text box based on the text box detected by [chinese_text_detection_db_mobile Module](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db_mobile&en_category=TextRecognition). Afterwards, the angle classification of the detected text box is performed. The final text recognition algorithm adopts CRNN (Convolutional Recurrent Neural Network), namely Convolutional Recurrent Neural Network. It is a combination of DCNN and RNN and is specialized for recognizing sequential objects in images. Used in conjunction with CTC loss, for text recognition, it can learn directly from text word-level or line-level annotations, and does not require detailed character-level annotations. This Module is an ultra-lightweight Chinese OCR model that supports direct prediction."

examples=[['test.png']]

gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="file"),title=title,description=description,examples=examples).launch(enable_queue=True,debug=True)