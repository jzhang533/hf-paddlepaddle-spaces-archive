import gradio as gr
import paddlehub as hub

senta = hub.Module(name="senta_bilstm")


def inference(text):
  test_text = [text]
  results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)
  for result in results:
    text_out = result['text']
    sent_label = result['sentiment_label']
    sent_key = result['sentiment_key']
    pos_prob = result['positive_probs']
    neg_prob = result['negative_probs']
  return text_out, sent_label, sent_key, pos_prob, neg_prob

  
title="senta_bilstm"
description="Sentiment Classification (Sentiment Classification, referred to as Senta) for Chinese texts with subjective descriptions, can automatically determine the sentiment polarity category of the text and give the corresponding confidence, which can help companies understand user consumption habits, analyze hot topics and crises Public opinion monitoring provides favorable decision support for enterprises. The model is based on a bidirectional LSTM structure, with sentiment types divided into positive and negative. The PaddleHub Module supports prediction and fine-tune."

examples=[['今天是个好日子']]
gr.Interface(inference,"text",[gr.outputs.Textbox(label="text"),gr.outputs.Textbox(label="sentiment label"),gr.outputs.Textbox(label="sentiment key"),gr.outputs.Textbox(label="positive probility"),gr.outputs.Textbox(label="negative probabillity")],title=title,description=description,examples=examples).launch(enable_queue=True)