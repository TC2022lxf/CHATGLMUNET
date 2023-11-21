from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
from run_predict import run_predict
tokenizer = AutoTokenizer.from_pretrained("ptuning/THUDM/chatglm-6b", trust_remote_code=True)
from Utils import load_model_on_gpus
model = load_model_on_gpus("ptuning/THUDM/chatglm-6b", num_gpus=2)
#model = AutoModel.from_pretrained("ptuning/THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
from run_predict import run_predict


"""Override Chatbot.postprocess"""


# def postprocess(self, y):
#     if y is None:
#         return []
#     for i, (message, response) in enumerate(y):
#         y[i] = (
#             None if message is None else mdtex2html.convert((message)),
#             None if response is None else mdtex2html.convert(response),
#         )
#     return y
#
#
# gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text
import json
def test_model(input_text,history):
    # 添加输入到对话历史中
    chat_history = [(input_text, "")]
    # 将输入文本转换为模型所需的张量格式
    inputs = "输入测试"
    inputs = "输入测试"
    # 使用模型生成回复
    outputs = "测试"
    # 将模型生成的回复转换为文本
    response = "回复测试"
    # 添加回复到对话历史中
    chat_history[-1] = (input_text, response)
    yield response, chat_history

def predict(input, chatbot, max_length, top_p, temperature, history):
    flag = 0
    if input != '':

        print(chatbot)
        if chatbot is None: begin()
        chatbot.append((parse_text(input), ""))
        flag = 0
    else:
        flag =1
        r_string, imgpath = run_predict(chatbot[-1][0][0])
        print(r_string, imgpath)
        input = '菌落国标是菌落总数大于30小于300.菌落信息如下：是否符合标准\n'+r_string
    print(input)
    print('flag值',flag)
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,temperature=temperature):
        input = parse_text(input)
        if flag == 0:
            chatbot[-1] = (parse_text(input), parse_text(response))
        else:
            if flag==1:
                chatbot[-1][-1] = (imgpath,)
                chatbot.append([parse_text(input), parse_text(response)])
                flag=2
            chatbot[-1][-1]=parse_text(response)
        print(flag)
        print(chatbot)
        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

def add_file(chatbot,file):
    chatbot = chatbot + [((file.name,), None)]
    print(file)
    return chatbot

def begin():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">ChatGLM</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            with gr.Column(scale=1, min_width=0):
                btn = gr.UploadButton("📁", file_types=["image"])

        history = gr.State([])

        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
        file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history])
    demo.queue().launch(share=False, inbrowser=True)

begin()