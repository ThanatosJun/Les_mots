import gradio as gr
from mermaid import generate_mermaid_quadrantChart
from PIL import Image
from io import BytesIO

def generate_mermaid_image(mermaid_code: str):
    # 獲取 Mermaid 圖表的二進制數據
    screenshot_bytes = generate_mermaid_quadrantChart(mermaid_code)
    
    # 利用 BytesIO 將圖片數據轉為可在 Gradio 中顯示的格式
    image = Image.open(BytesIO(screenshot_bytes))
    image.seek(0)  # 重設讀取指標
    return image

# Mermaid 語法示例
# mermaid_code = """
# quadrantChart
#     title Reach and engagement of campaigns
#     x-axis Low Reach --> High Reach
#     y-axis Low Engagement --> High Engagement
#     quadrant-1 We should expand
#     quadrant-2 Need to promote
#     quadrant-3 Re-evaluate
#     quadrant-4 May be improved
#     Campaign A: [0.3, 0.6]
#     Campaign B: [0.45, 0.23]
#     Campaign C: [0.57, 0.69]
#     Campaign D: [0.78, 0.34]
#     Campaign E: [0.40, 0.34]
#     Campaign F: [0.35, 0.78]
# """
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
class LLM():
    def __init__():
        llm = OllamaLLM(model='llama3.2')
    def question_request(user_input :str):
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
            ('user', 'Question: {input}'),
        ])
        

class GradioInterface():
    # 建立 Gradio 接口
    def __init__(self):
        user_input = []
        llm = LLM()
        self.ControllerView = gr.Interface(fn=llm.question_request, 
                            inputs=gr.Textbox(inputs=user_input, label="Mermaid Code"),
                            outputs=[
                                gr.Textbox(label="LLM Text Response"),
                                gr.Image(type="pil", label="Mermaid Diagram")
                                ]
                            )
    def launch(self):
        self.ControllerView.launch()