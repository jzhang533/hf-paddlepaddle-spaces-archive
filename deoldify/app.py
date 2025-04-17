import os
os.system("hub install deoldify==1.0.1")
import gradio as gr
import paddlehub as hub
from pathlib import Path


model = hub.Module(name='deoldify')

def inference(image):
    model.predict(image.name)
    return './output/DeOldify/'+Path(image.name).stem+".png"

title = "DeOldify"
description = "Gradio demo for DeOldify. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/jantic/DeOldify' target='_blank'>Github Repo</a></p>"

examples=[['lunch.jpeg']]
iface = gr.Interface(inference, inputs=gr.inputs.Image(type="filepath"), outputs=gr.outputs.Image(type="filepath"),examples=examples,enable_queue=True,title=title,article=article,description=description)
iface.launch()