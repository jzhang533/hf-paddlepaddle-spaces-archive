import gradio as gr
import paddlehub as hub


transformer_zh = hub.Module(name="transformer_zh-en")

def inference(text):
  results = transformer_zh.predict(data=[text])
  return results[0]

  
title="transformer_zh-en"
description="Transformer model used for translating Chinese into English."

examples=[['今天是个好日子']]
gr.Interface(inference,"text",[gr.outputs.Textbox(label="Translation")],title=title,description=description,examples=examples).launch(enable_queue=True)