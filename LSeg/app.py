import cv2
import paddle
import paddlehub as hub
import gradio as gr


module = hub.Module(name="lseg")


def segment(image, labels):
    try:
        long_size = max(image.shape[:2])
        if long_size > 512:
            f = 512 / long_size
            image = cv2.resize(image, (0, 0), fx=f, fy=f)
        results = module.segment(
            image=image[..., ::-1],
            labels=[item for item in labels.split('\n') if item != '']
        )

        return [
            results['color'][..., ::-1],
            results['mix'][..., ::-1],
            *[cv2.cvtColor(v, cv2.COLOR_BGRA2RGBA) for v in results['classes'].values()]
        ]
    except:
        paddle.disable_static()
        raise ValueError


gr.Interface(
    title='LSeg: Language-driven Semantic Segmentation',
    fn=segment,
    inputs=[
        gr.Image(),
        gr.Textbox(placeholder='other\ncat', lines=5, max_lines=50),
    ],
    outputs=[
        gr.Gallery().style(grid=[2, 3], height="auto")
    ],
    article='''## More
* There are more interesting models in [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), you can star [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) to follow.

* Besides, you can use free GPU resourses in [AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4580391) to enjoy more cases, have fun.
    
    [![](https://user-images.githubusercontent.com/22424850/187849103-074cb6d2-a9b4-49a1-b1f0-fc130049769f.png)](https://github.com/PaddlePaddle/PaddleHub/stargazers)

## References
* Paper: [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546)

* Offical Code: [isl-org/lang-seg](https://github.com/isl-org/lang-seg)
''',
    examples=[['cat.jpeg', 'other\ncat']],
    cache_examples=True
).launch()
