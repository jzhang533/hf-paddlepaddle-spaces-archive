#-*- coding: UTF-8 -*-
# Copyright 2022 the HuggingFace Team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import base64

import gradio as gr
import cv2

from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser


doc_parser = DocParser()
task_instance = Taskflow(
    "information_extraction",
    model="uie-x-base",
    task_path="PaddlePaddle/uie-x-base",
    from_hf_hub=True)

examples = [
    [
        "business_card.png",
        "Name;Title;Web Link;Email;Address",
    ],
    [
        "license.jpeg",
        "Name;DOB;ISS;EXP",
    ],
    [
        "statements.png",
        "Date|Gross profit",
    ],    
    [
        "invoice.jpeg",
        "名称;纳税人识别号;开票日期",
    ],
    [
        "custom.jpeg",
        "收发货人;进口口岸;进口日期;运输方式;征免性质;境内目的地;运输工具名称;包装种类;件数;合同协议号"
    ],
    [
        "resume.png",
        "职位;年龄;学校|时间;学校|专业",
    ],
]

example_files = {
    "Name;Title;Web Link;Email;Address": "business_card.png",
    "Name;DOB;ISS;EXP": "license.jpeg",
    "Date|Gross profit": "statements.png",
    "职位;年龄;学校|时间;学校|专业": "resume.png",
    "收发货人;进口口岸;进口日期;运输方式;征免性质;境内目的地;运输工具名称;包装种类;件数;合同协议号": "custom.jpeg",
    "名称;纳税人识别号;开票日期": "invoice.jpeg",
}

lang_map = {
    "resume.png": "ch",
    "custom.jpeg": "ch",
    "business_card.png": "en",
    "invoice.jpeg": "ch",
    "license.jpeg": "en",
    "statements.png": "en",
}

def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def BGR2RGB(img):
    pilimg = img.copy()
    pilimg[:, :, 0] = img[:, :, 2]
    pilimg[:, :, 2] = img[:, :, 0]
    return pilimg


def np2base64(image_np):
    image_np = BGR2RGB(image_np)
    image = cv2.imencode('.jpg', image_np)[1]
    base64_str = str(base64.b64encode(image))[2:-1]
    return base64_str


def process_path(path):
    error = None
    if path:
        try:
            if path.endswith(".pdf"):
                images_list = [doc_parser.read_pdf(path)]
            else:
                images_list = [doc_parser.read_image(path)]
            return (
                path,
                gr.update(visible=True, value=images_list),
                gr.update(visible=True),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                None,
            )
        except Exception as e:
            traceback.print_exc()
            error = str(e)
    return (
        None,
        gr.update(visible=False, value=None),
        gr.update(visible=False),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=error) if error is not None else None,
        None,
    )


def process_upload(file):
    if file:
        return process_path(file.name)
    else:
        return (
            None,
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            None,
        )

def get_schema(schema_str):
    def _is_ch(s):
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                return True
        return False
    schema_lang = "ch" if _is_ch(schema_str) else "en"
    schema = schema_str.split(";")
    schema_list = []
    for s in schema:
        cand = s.split("|")
        if len(cand) == 1:
            schema_list.append(cand[0])
        else:
            subject = cand[0]
            relations = cand[1:]
            added = False
            for a in schema_list:
                if isinstance(a, dict):
                    if subject in a.keys():
                        a[subject].extend(relations)
                        added = True
                        break
            if not added:
                a = {subject: relations}
                schema_list.append(a)
    return schema_list, schema_lang


def run_taskflow(document, schema, argument):
    task_instance.set_schema(schema)
    task_instance.set_argument(argument)
    return task_instance({'doc': document})


def process_doc(document, schema, ocr_lang, layout_analysis):
    if [document, schema] in examples:
        ocr_lang = lang_map[document]

    if not schema:
        schema = '时间;组织机构;人物'
    if document is None:
        return None, None

    layout_analysis = True if layout_analysis == "yes" else False
    schema, schema_lang = get_schema(dbc2sbc(schema))
    argument = {
        "ocr_lang": ocr_lang,
        "schema_lang": schema_lang,
        "layout_analysis": layout_analysis
    }
    prediction = run_taskflow(document, schema, argument)[0]

    if document.endswith(".pdf"):
        _image = doc_parser.read_pdf(document)
    else:
        _image = doc_parser.read_image(document)

    img_show = doc_parser.write_image_with_results(
        np2base64(_image),
        result=prediction,
        return_image=True)
    img_list = [img_show]

    return (
        gr.update(visible=True, value=img_list),
        gr.update(visible=True, value=prediction),
    )


def load_example_document(img, schema, ocr_lang, layout_analysis):
    if img is not None:
        document = example_files[schema]
        choice = lang_map[document].split("-")
        ocr_lang = choice[0]
        preview, answer = process_doc(document, schema, ocr_lang, layout_analysis)
        return document, schema, preview, gr.update(visible=True), answer
    else:
        return None, None, None, gr.update(visible=False), None


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


with gr.Blocks() as demo:
    gr.HTML(read_content("header.html"))
    gr.Markdown(
        "Open-sourced by [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP), **UIE-X** is a universal information extraction engine for both scanned document and text inputs. It supports Entity Extraction, Relation Extraction and Event Extraction tasks. "
        "UIE-X performs well on a zero-shot settings, which is enabled by a flexible schema that allows you to specify extraction targets with simple natural language. "
        "Moreover, on [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP), we provide a comprehensive and easy-to-use fine-tuning and few-shot customization workflow. <br>" 
        "Want to dive deeper? Check out our [AIStudio Notebook](https://aistudio.baidu.com/aistudio/projectdetail/5261592) and [Colab Notebook](https://colab.research.google.com/drive/1ZY_ELZgoemJNoa6baWpgtzebLgoCT8MK?usp=sharing). "
        "For more details, please visit our [GitHub](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/README_en.md)"
    )

    document = gr.Variable()
    is_text = gr.Variable()
    example_schema = gr.Textbox(visible=False)
    example_image = gr.Image(visible=False)
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## 1. Select a file 选择文件", elem_id="select-a-file")
                img_clear_button = gr.Button(
                    "Clear", variant="secondary", elem_id="file-clear", visible=False
                )
            image = gr.Gallery(visible=False)
            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Row():
                        url = gr.Textbox(
                            show_label=False,
                            placeholder="URL",
                            lines=1,
                            max_lines=1,
                            elem_id="url-textbox",
                        )
                        submit = gr.Button("Get")
                    url_error = gr.Textbox(
                        visible=False,
                        elem_id="url-error",
                        max_lines=1,
                        interactive=False,
                        label="Error",
                    )
            gr.Markdown("## <center> — or — </center>")
            upload = gr.File(label=None, interactive=True, elem_id="short-upload-box")
            gr.Examples(
                examples=examples,
                inputs=[example_image, example_schema],
            )

        with gr.Column():
            gr.Markdown("## 2. Information Extraction 信息抽取 ")
            gr.Markdown("### 👉 Set a schema 设置schema")
            gr.Markdown("Entity extraction: entity type should be separated by ';', e.g. **Person;Organization**")
            gr.Markdown("实体抽取：实体类别之间以';'分割，例如 **人物；组织机构**")
            gr.Markdown("Relation extraction: set the subject and relation type, separated by '|', e.g. **Person|Date;Person|Email**")
            gr.Markdown("关系抽取：需配置主体和关系类别，中间以'|'分割，例如 **人物|出生时间；人物|邮箱**")
            gr.Markdown("### 👉 Model customization 模型定制")
            gr.Markdown("We recommend to further improve the extraction performance in specific domain through the process of [data annotation & fine-tuning](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/document/README_en.md)")
            gr.Markdown("我们建议通过[数据标注+微调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/document/README_en.md)的流程进一步增强模型在特定场景的效果")

            schema = gr.Textbox(
                label="Schema",
                placeholder="e.g. Name|Company;Name|Position;Email;Phone Number",
                lines=1,
                max_lines=1,
            )

            ocr_lang = gr.Radio(
                choices=["ch", "en"],
                value="en",
                label="OCR语言 / OCR Language (Please choose ch for Chinese images.)",
            )

            layout_analysis = gr.Radio(
                choices=["yes", "no"],
                value="no",
                label="版面分析 / Layout analysis (Better extraction for multi-line text)",
            )

            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button(
                    "Submit", variant="primary", elem_id="submit-button"
                )          
            with gr.Column():
                output = gr.JSON(label="Output", visible=False)

    for cb in [img_clear_button, clear_button]:
        cb.click(
            lambda _: (
                gr.update(visible=False, value=None),
                None,
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None,
                None,
                None,
                gr.update(visible=False, value=None),
                None,
            ),
            inputs=clear_button,
            outputs=[
                image,
                document,
                output,
                img_clear_button,
                example_image,
                upload,
                url,
                url_error,
                schema,
            ],
        )

    upload.change(
        fn=process_upload,
        inputs=[upload],
        outputs=[document, image, img_clear_button, output, url_error],
    )
    submit.click(
        fn=process_path,
        inputs=[url],
        outputs=[document, image, img_clear_button, output, url_error],
    )

    schema.submit(
        fn=process_doc,
        inputs=[document, schema, ocr_lang, layout_analysis],
        outputs=[image, output],
    )

    submit_button.click(
        fn=process_doc,
        inputs=[document, schema, ocr_lang, layout_analysis],
        outputs=[image, output],
    )

    example_image.change(
        fn=load_example_document,
        inputs=[example_image, example_schema, ocr_lang, layout_analysis],
        outputs=[document, schema, image, img_clear_button, output],
    )

    gr.HTML(read_content("footer.html"))


if __name__ == "__main__":
    demo.queue().launch()
