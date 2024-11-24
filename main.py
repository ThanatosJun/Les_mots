import gradio as gr
from mermaid import generate_mermaid_quadrantChart
from PIL import Image
from io import BytesIO

# def generate_mermaid_image(mermaid_code: str):
#     # 獲取 Mermaid 圖表的二進制數據
#     screenshot_bytes = generate_mermaid_quadrantChart(mermaid_code)
    
#     # 利用 BytesIO 將圖片數據轉為可在 Gradio 中顯示的格式
#     image = Image.open(BytesIO(screenshot_bytes))
#     image.seek(0)  # 重設讀取指標
#     return image

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
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
class LLM():
    def __init__(self):
        llm = OllamaLLM(model='llama3.2')
        text_prompt = ChatPromptTemplate.from_messages([
            ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
            ('user', 'Question: {input}'),
        ])
        self.code_prompt = ChatPromptTemplate.from_messages([
            ('system', 'Generate a mermaid syntax diagram based on the following input.'),
            ('user', '{input}'),
        ])
        FAISS_vectordb = "faiss_index"
        vectordb = FAISS.load_local(FAISS_vectordb, embeddings=None)
        print(f"Finish FAISS Load")
        retriever = vectordb.as_retriever()
        self.document_chain = create_stuff_documents_chain(llm, text_prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, self.document_chain)
    def text_request(self ,user_input :str):
        RAG_context = []
        text_response = self.retrieval_chain.invoke({
        'input': user_input,
        'context': RAG_context
        })
        print(f"Finish Text Request\ntext_response = {text_response}")
        return text_response
    def mermaid_code_request(self, llm_text):
        code_response = self.code_prompt.format_messages(input = llm_text)
        print(f"Finish Code Request\ncode_response = {code_response}")
        return code_response
    def mermaid_diagram_request(self, llm_code):
        screenshot_bytes = generate_mermaid_quadrantChart(llm_code)
        # 利用 BytesIO 將圖片數據轉為可在 Gradio 中顯示的格式
        mermaid_diagram = Image.open(BytesIO(screenshot_bytes))
        mermaid_diagram.seek(0)  # 重設讀取指標
        print(f"Finish Mermaid Diagram")
        return mermaid_diagram
    def question_request(self, user_input):
        text_response = self.text_request(user_input)
        mermaid_code = self.mermaid_code_request(text_response)
        mermaid_diagram = self.mermaid_diagram_request(mermaid_code)
        print(f"Finish Answer")
        return text_response, mermaid_diagram


class GradioUI():
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

if __name__ == "__main__":
    webUI = GradioUI()
    webUI.launch()
