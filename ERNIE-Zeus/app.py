import gradio as gr
import paddlehub as hub

ernie_zeus = hub.Module(name='ernie_zeus')


def inference(task: str,
              text: str,
              min_dec_len: int = 2,
              seq_len: int = 512,
              topp: float = 0.9,
              penalty_score: float = 1.0):

    func = getattr(ernie_zeus, task)
    try:
        result = func(text, min_dec_len, seq_len, topp, penalty_score)
        return result
    except Exception as error:
        return str(error)


title = "ERNIE-Zeus"

description = "ERNIE-Zeus model, which supports Chinese text generates task."

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }

        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
        'text_summarization',
        '外媒7月18日报道，阿联酋政府当日证实该国将建设首个核电站，以应对不断上涨的用电需求。分析称阿联酋作为世界第三大石油出口国，更愿意将该能源用于出口，而非发电。首座核反应堆预计在2017年运行。cntv李婉然编译报道',
        4, 512, 0.3, 1.0
    ],
    [
        'copywriting_generation',
        '芍药香氛的沐浴乳',
        8, 512, 0.9, 1.2
    ],
    [
        'novel_continuation',
        '昆仑山可以说是天下龙脉的根源，所有的山脉都可以看作是昆仑的分支。这些分出来的枝枝杈杈，都可以看作是一条条独立的龙脉。',
        2, 512, 0.9, 1.2
    ],
    [
        'answer_generation',
        '做生意的基本原则是什么？',
        2, 512, 0.5, 1.2
    ],
    [
        'couplet_continuation',
        '天增岁月人增寿',
        2, 512, 1.0, 1.0
    ],
    [
        'composition_generation',
        '拔河比赛',
        128, 512, 0.9, 1.2
    ],
    [
        'text_cloze',
        '她有着一双[MASK]的眼眸。',
        1, 512, 0.3, 1.2
    ],
]

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                  margin-bottom: 10px;
                  justify-content: center;
                "
              >
              <img src="https://user-images.githubusercontent.com/22424850/187387422-f6c9ccab-7fda-416e-a24d-7d6084c46f67.jpg" alt="Paddlehub" width="40%">
              </div> 
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                  margin-bottom: 10px;
                  justify-content: center;
                ">
              <h1 style="font-weight: 900; margin-bottom: 7px;">
                  ERNIE-Zeus Demo
              </h1>
              </div> 
              <p style="margin-bottom: 10px; font-size: 94%">
                ERNIE-Zeus is a state-of-the-art Chinese text generates model.
              </p>
            </div>
        """
    )
    with gr.Blocks():
        text = gr.Textbox(
            label="input_text",
            placeholder="Please enter Chinese text.",
        )
        task = gr.Dropdown(label="task",
                           choices=[
                               'text_summarization',
                               'copywriting_generation',
                               'novel_continuation',
                               'answer_generation',
                               'couplet_continuation',
                               'composition_generation',
                               'text_cloze'
                           ],
                           value='text_summarization')

        min_dec_len = gr.Slider(
            minimum=1, maximum=511, value=1, label="min_dec_len", step=1, interactive=True)
        seq_len = gr.Slider(minimum=2, maximum=512, value=128,
                            label="seq_len", step=1, interactive=True)
        topp = gr.Slider(minimum=0.0, maximum=1.0, value=1.0,
                         label="topp", step=0.01, interactive=True)
        penalty_score = gr.Slider(
            minimum=1.0, maximum=2.0, value=1.0, label="penalty_score", step=0.01, interactive=True)

        text_gen = gr.Textbox(label="generated_text")
        btn = gr.Button(value="Generate text")

        ex = gr.Examples(examples=examples, fn=inference, inputs=[
                         task, text, min_dec_len, seq_len, topp, penalty_score], outputs=text_gen, cache_examples=False)

        text.submit(inference, inputs=[
                    task, text, min_dec_len, seq_len, topp, penalty_score], outputs=text_gen)
        btn.click(inference, inputs=[
                  task, text, min_dec_len, seq_len, topp, penalty_score], outputs=text_gen)
        gr.Markdown(
            '''     
## More
* There are more interesting models in [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), you can star [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) to follow.
* Besides, you can use free GPU resourses in [AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4462918) to enjoy more cases, have fun.
    [![](https://user-images.githubusercontent.com/22424850/187849103-074cb6d2-a9b4-49a1-b1f0-fc130049769f.png)](https://github.com/PaddlePaddle/PaddleHub/stargazers)
            '''
        )
        gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://github.com/PaddlePaddle/PaddleHub" style="text-decoration: underline;" target="_blank">PaddleHub</a> and <a href="https://wenxin.baidu.com" style="text-decoration: underline;" target="_blank">文心大模型</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
                </div>
           """
        )

block.queue(max_size=100000, concurrency_count=100000).launch()
