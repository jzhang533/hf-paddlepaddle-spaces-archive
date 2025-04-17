import gradio as gr
import os

def inference(text):
    os.system("paddlespeech tts --input '"+text+"' --output output.wav")
    return "output.wav"

title = "PaddleSpeech TTS"

description = "Gradio demo for PaddleSpeech: A Speech Toolkit based on PaddlePaddle for TTS. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://github.com/PaddlePaddle/PaddleSpeech' target='_blank'>Github Repo</a></p>"

examples=[['你好，欢迎使用百度飞桨深度学习框架！']]

gr.Interface(
    fn=inference, 
    inputs=gr.Textbox(label="input text", lines=10), 
    outputs=gr.Audio(type="filepath", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples
).launch(debug=True)