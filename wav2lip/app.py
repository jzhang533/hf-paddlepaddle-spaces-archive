import os
os.system("pip install gradio==2.7.5.2")
os.system("pip install paddlepaddle==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")
os.system("hub install wav2lip==1.0.0")
os.system("pip install -r requirements.txt")
import gradio as gr
import paddlehub as hub

module = hub.Module(name="wav2lip")

def inference(image,audio):
    module.wav2lip_transfer(face=image, audio=audio, output_dir='.', use_gpu=False)  
    return "result.mp4"
    
title = "Wav2lip"
description = "Gradio demo for Wav2lip: Accurately Lip-syncing Videos In The Wild. To use it, simply upload your image and audio file, or click one of the examples to load them. Read more at the links below. Please trim audio file to maximum of 3-4 seconds"

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2008.10010' target='_blank'>A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild</a> | <a href='https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/wav2lip.md' target='_blank'>Github Repo</a></p>"
examples=[['monatest.jpeg',"game.wav"]]
iface = gr.Interface(inference, [gr.inputs.Image(type="filepath"),gr.inputs.Audio(source="microphone", type="filepath")], 
outputs=gr.outputs.Video(label="Output Video"),examples=examples,title=title,article=article,description=description)
iface.launch(cache_examples=True,enable_queue=True)