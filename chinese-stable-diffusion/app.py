import os
import requests
from uuid import uuid4
from PIL import Image
import gradio as gr


def gen_image(prompt: str):
    """generate the image from the chinese stable diffusion model of paddlenlp server

    Args:
        prompt (str): the source of the prompt
    """
    if not prompt:
        return
    access_token = os.environ['token']

    url = f"https://aip.baidubce.com/rpc/2.0/nlp-itec/poc/chinese_stable_diffusion?access_token={access_token}"

    content = requests.post(url, json={"text": prompt}).content
    try:
        new_content = content.decode(encoding='utf-8')

        if new_content.startswith("error: "):
            return []
    except:
        pass

    cache_dir = 'images'
    os.makedirs(cache_dir, exist_ok=True)

    tempfile = os.path.join(cache_dir, f'{str(uuid4())}.png')
    with open(tempfile, 'wb') as f:
        f.write(content)
    image = Image.open(tempfile)
    os.remove(tempfile)
    return [image]

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

block = gr.Blocks(css=read_content('style.css'))

examples = [
    '枯藤老树昏鸦，水墨画',
    '小桥流水人家，水墨画',
    '古道西风瘦马，水墨画',
    '夕阳西下，水墨画',
    '断肠人在天涯，水墨画',
    '白日依山尽，水墨画',
    '黄河入海流，水墨画',
    '欲穷千里目，水墨画',
    '更上一层楼，水墨画',
]


with block:
    gr.HTML(read_content("header.html"))
    gr.Markdown("> 非常抱歉，由于一些原因，此Space临时停止服务，等国庆之后会重新发布。", elem_id='warning', visible=True)
    gr.Markdown("[![Stargazers repo roster for @PaddlePaddle/PaddleNLP](https://reporoster.com/stars/PaddlePaddle/PaddleNLP)](https://github.com/PaddlePaddle/PaddleNLP)")
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="输入中文，生成图片",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )

                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[1, 1], height="auto")

        gr.Examples(examples=examples, fn=gen_image, inputs=text, outputs=gallery)
        # text.submit(gen_image, inputs=text, outputs=gallery)
        # btn.click(gen_image, inputs=text, outputs=gallery)

    gr.Image('./paddlenlp-preview.jpeg')
    gr.HTML(read_content("footer.html"))

block.queue(concurrency_count=5).launch(debug=True)