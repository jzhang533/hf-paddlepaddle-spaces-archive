import gradio as gr
import paddlehub as hub

model = hub.Module(name='SkyAR')


def inference(invid):
  result = model.MagicSky(
    video_path=[invid],
    save_path=['out.mp4']
  )
  return 'out.mp4'

  
title="SkyAR"
description="SkyAR is a visual method for sky replacement and coordination in video, which mainly consists of three cores: sky matting network, motion estimation and image fusion."

examples=[['rabbit.jpeg']]
gr.Interface(inference,gr.inputs.Video(type="mp4"),"video",title=title,description=description,examples=examples).launch(enable_queue=True)