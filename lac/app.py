import gradio as gr
import paddlehub as hub


lac = hub.Module(name="lac")

def inference(text):
  text = [text]
  results = lac.cut(text=text, use_gpu=False, batch_size=1, return_tag=True)
  for result in results:
    word = result['word']
    tag = result['tag']
  return word, tag

  
title="LAC"
description="Lexical Analysis of Chinese, or LAC for short, is a joint lexical analysis model that can comprehensively complete Chinese word segmentation, part-of-speech tagging, and proper name recognition tasks. Evaluation on Baidu's self-built data set, LAC effect: Precision=88.0%, Recall=88.7%, F1-Score=88.4%. The PaddleHub Module supports prediction."

examples=[['今天是个好日子']]
gr.Interface(inference,"text",[gr.outputs.Textbox(label="word"),gr.outputs.Textbox(label="tag")],title=title,description=description,examples=examples).launch(enable_queue=True)