import gradio as gr
from mermaid import generate_mermaid_quadrantChart
from PIL import Image
from io import BytesIO

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
class LLM():
    def __init__(self):
        llm = OllamaLLM(model='llama3.2')
        embeddings = OllamaEmbeddings(model="llama3.2")
        text_prompt = ChatPromptTemplate.from_messages([
            ('system', 
                """You will act as an expert in French, skilled at analyzing the semantics, grammatical properties, and contextual applications of French words. Based on a given word, you will identify 10 synonyms or near-synonyms with the same grammatical category. These will be classified and explained according to their emotional intensity and formality.
                Please adhere to the following rules for your response:
                Provide two paragraphs in total:
                0.Please use zh-TW 繁體中文 to response.
                1.First paragraph: List the "replaced word"(原單詞) and the emotional intensity and formality of the 5 "replacement words"(替代詞).
                    ex:
                    Word/Emotional Intensity/Formality
                    原單詞：colère 情感強度：3 正式性：3
                    替代詞：
                    1. mécontentement 情感強度：1 正式性：2
                    2. agacement 情感強度：2 正式性：3
                    3. indignation 情感強度：3 正式性：4
                    4. courroux 情感強度：4 正式性：5
                    5. rage 情感強度：5 正式性：1
                2.Second paragraph: Provide a brief explanation of the specific meaning and usage context for each "replacement word."
                    ex:
                    簡短說明：
                    1. mécontentement: 表示輕微的不滿或失望。
                    2. agacement: 輕度煩躁的情感反應。
                    3. indignation: 對不公平或不道德行為的憤慨。
                    4. courroux: 嚴肅且強烈的憤怒。
                    5. rage: 極度憤怒。
                3.The replaced word must be the same grammatical groups as the word entered by the user.
                    ex: input adjective word and answer adjective words. 
                4.For each "replacement word," include:
                    a.Emotional intensity: Rated from 1 to 5 (1 being the lowest, 5 being the highest), indicating the strength of emotional expression.
                    b.The emotional intensity and formality of the replaced word are fixed at 3 (medium level).
                    c.Ensure diversity by covering all intensity and formality scores (i.e., 1 through 5).
                    d.Only provide synonyms and avoid antonyms.
                    e.Explanations should be concise and highlight differences in emotional intensity and formality in context.
                    f.Arrange the replacement words from lowest to highest emotional intensity.
                    g.Do not repeat replacement words.
                    h.Ensure the order of explanation matches the order of the listed replacement words."""),
            ('user','Here is an example of the expected input:Please list 5 synonyms for "colère.'),
            ('system',"""Here is an example of the expected output:
                單詞/情緒強度/正式性
                原單詞：colère 情感強度：3 正式性：3
                替代詞：
                1. mécontentement 情感強度：1 正式性：2
                2. agacement 情感強度：2 正式性：3
                3. indignation 情感強度：3 正式性：4
                4. courroux 情感強度：4 正式性：5
                5. rage 情感強度：5 正式性：1

                簡短說明：
                1. mécontentement: 表示輕微的不滿或失望。
                2. agacement: 輕度煩躁的情感反應。
                3. indignation: 對不公平或不道德行為的憤慨。
                4. courroux: 嚴肅且強烈的憤怒。
                5. rage: 極度憤怒。"""),
            ('system', 'Answer the user\'s questions in zh-TW, based on the context provided below:\n\n{context}'),
            ('user', 'Question: {input}'),
        ])
        code_prompt = ChatPromptTemplate.from_messages([
            ('system', """Generate mermaid code of quadrantChart diagram based on the following replacement words (替代詞).
                Instructions:
                1. Respond **only** with valid Mermaid code. Do not include explanations, greetings, or any additional text.
                2. The diagram should use the input data to compute coordinates as follows:
                    - Emotional Intensity = 情感強度 / 5
                    - Formality = 正式性 / 5
                3. Coordinates must be expressed as [情感強度/5, 正式性/5] (not as tuples or in any other format).
                4. Ensure all values are normalized to be between 0 and 1.
                5. Use the example format strictly, including title, axis labels, quadrant labels, and proper data representation."""),
            ('user', """Here is an example of the expected input:
                原單詞：colère 情感強度：3 正式性：3
                替代詞：
                1. mécontentement 情感強度：1 正式性：2
                2. agacement 情感強度：2 正式性：3
                3. indignation 情感強度：3 正式性：4
                4. courroux 情感強度：4 正式性：5
                5. rage 情感強度：5 正式性：1"""),
            ('system', """Here is an example of the expected output:
                quadrantChart
                    title Emotional Intensity and Formality
                    x-axis Low Emotional Intensity --> High Emotional Intensity
                    y-axis Low Formality --> High Formality
                    quadrant-1 High Emotion, High Formality
                    quadrant-2 Low Emotion, High Formality
                    quadrant-3 Low Emotion, Low Formality
                    quadrant-4 High Emotion, Low Formality
                    colere: [0.6, 0.6]
                    mecontentement: [0.2, 0.4]
                    agacement: [0.4, 0.6]
                    indignation: [0.6, 0.8]
                    courroux: [0.8, 1]
                    rage: [1, 0.2]"""),
            ('user', 'Please use following imformation to generate mermaid code of quadrantChart:{input}'),
        ])
        FAISS_vectordb = "faiss_index"
        vectordb = FAISS.load_local(
            FAISS_vectordb,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Finish FAISS Load")
        retriever = vectordb.as_retriever()
        self.document_chain = create_stuff_documents_chain(llm, text_prompt)
        self.document_diagram_chain = code_prompt | llm
        self.retrieval_chain = create_retrieval_chain(retriever, self.document_chain)
    
    def text_request(self ,user_input :str):
        RAG_context = []
        text_response = self.retrieval_chain.invoke({
            'input': user_input,
            'context': RAG_context
        })['answer']
        print(f"Finish Text Request\ntext_response = {text_response}")
        return text_response

    def mermaid_code_request(self, llm_text):
        code_response = self.document_diagram_chain.invoke({'input': f"\"\"\"{llm_text}\"\"\""})
        print(f"Finish Code Request\ncode_response = {code_response}")
        return code_response

    def mermaid_diagram_request(self, llm_code):
        screenshot_bytes = generate_mermaid_quadrantChart(llm_code)
        # 利用 BytesIO 將圖片數據轉為可在 Gradio 中顯示的格式
        mermaid_diagram = Image.open(BytesIO(screenshot_bytes))
        mermaid_diagram.seek(0)  # 重設讀取指標
        print(f"Finish Mermaid Diagram")
        return mermaid_diagram
    
    def clean_mermaid_code(self, output: str) -> str:
        if output.startswith("```mermaid"):
            output = output[len("```mermaid"):].strip()
        if output.endswith("```"):
            output = output[:-len("```")].strip()
        return output

    def question_request(self, user_input):
        text_response = self.text_request(user_input)
        mermaid_code = self.mermaid_code_request(text_response)
        mermaid_code = self.clean_mermaid_code(mermaid_code)
        print(f"clean-code = \n{mermaid_code}")
        mermaid_diagram = self.mermaid_diagram_request(mermaid_code)
        print(f"Finish Answer")
        return text_response, mermaid_diagram


class GradioUI():
    # 建立 Gradio 接口
    def __init__(self):
        user_input = []
        llm = LLM()
        self.ControllerView = gr.Interface(
            fn=llm.question_request, 
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
