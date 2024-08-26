# build a UI for mingraphrag with gradio
import gradio as gr
from main import ask,load_init_data

embedding_map = load_init_data('./data-mingraphrag/')

def ask_with_UI(input_text):
    for response in ask(input_text,embedding_map):  # 逐步返回生成器的输出
        yield response 

iface = gr.Interface(
    fn=ask_with_UI,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Output"),
    title="Mingraphrag Text Analyzer",
    description="Graphrag example with a simple story",
)

iface.launch()
