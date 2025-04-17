import gradio as gr
import paddlehub as hub


jieba_paddle = hub.Module(name="jieba_paddle")

def inference(text):
  results = jieba_paddle.cut(sentence=text)
  return results

  
title="jieba_paddle"
description="jieba_paddle is a word segmentation model based on paddlepaddle deep learning framework."

examples=[['今天是个好日子']]
gr.Interface(inference,"text",[gr.outputs.Textbox(label="words")],title=title,description=description,examples=examples).launch(enable_queue=True)