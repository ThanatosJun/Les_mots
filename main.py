import gradio as gr
from mermaid import generate_mermaid_quadrantChart
from PIL import Image
from io import BytesIO
from unidecode import unidecode

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
                #  and classify them based on emotional intensity and formality.
class LLM():
    def __init__(self):
        llm = OllamaLLM(model='llama3.2')
        embeddings = OllamaEmbeddings(model="llama3.2")
        text_prompt = ChatPromptTemplate.from_messages([
            ('system', 
                """You will act as an expert in French, skilled at analyzing the French semantics, grammatical properties, and contextual applications of French words. Based on a given word, you will identify synonyms or near-synonyms with the same grammatical category.
                Task Description:
                1. Your goal is to analyze a given French word, identify its synonyms (同義詞) and defind them clearly.
                2. You will identify 5 synonyms with the same grammatical category as the given word.
                3. Use zh-TW 繁體中文 to respond all the questions."""),
            ('system', """
                Response Format:
                簡短說明：
                colère:通常指對不公平或冒犯行為感到不滿或憤慨時產生的怒氣。
                1. mécontentement: 表示輕微的不滿或失望。
                2. agacement: 輕度煩躁的情感反應。
                3. indignation: 對不公平或不道德行為的憤慨。
                4. courroux: 嚴肅且強烈的憤怒。
                5. rage: 極度憤怒。"""),
            ('system', """
                Additional Rules:
                - Use only synonyms. Avoid antonyms or unrelated words.
                - Arrange synonyms in ascending order of emotional intensity.
                - Provide concise and accurate explanations.
                - Do not repeat replacement words."""),
            ('system', """
                <root> ::= <interview> | <interview> <interview>
                <replacement> ::= <word> ": " <description>
                <word> ::= /* 可填入任何自定義單詞 */
                <description> ::= /* 可填入其他情緒的描述文字 */
            """),
            ('user', 'With zh-TW and based on the context below :\n\n{context}\n to answer the question directly:{input}'),
        ])
        emotion_prompt = ChatPromptTemplate.from_messages([
            ('system', """Your goal is to classify given datas into emotional intensity and formality range.
                Instruction:
                - Depending on the definding and words to Classify their emotional intensity and formality correctly from 1 o 5.
                - Ensure diversity in emotional intensity and formality, covering all levels (1 to 5).
                - Emotional intensity and formality range from 1 (lowest) to 5 (highest).
                - Respond **only** with emtional intensity and formality.
                - Use the example format strictly, including words, 情感強度, 正式性, and proper value"""),
            ('system', """Response Format:
                colère: 情感強度：3 正式性：3
                替代詞：
                1. mécontentement: 情感強度：1 正式性：2
                2. agacement: 情感強度：2 正式性：3
                3. indignation: 情感強度：3 正式性：4
                4. courroux: 情感強度：4 正式性：5
                5. rage: 情感強度：5 正式性：1"""),
            ('system', """
                <root> ::= <primary_emotion> | <synonym_list>
                <primary_emotion> ::= <word> <intensity_formality>
                <synonym_list> ::= <synonym> | <synonym> <synonym_list>
                <synonym> ::= <word> <intensity_formality>
                <word> ::= /* 可以填入任何自定義單詞 */
                <intensity_formality> ::= "情感強度：" <intensity> " 正式性：" <formality>
                <intensity> ::= "1" | "2" | "3" | "4" | "5"
                <formality> ::= "1" | "2" | "3" | "4" | "5"
            """),
            ('user', "seperate from 1 to 5 in intensity and formality range:{input}")
        ])

        code_prompt = ChatPromptTemplate.from_messages([
            ('system', """Generate mermaid code of quadrantChart diagram based on the following replacement words (替代詞).
                Instruction:
                - Respond **only** with valid Mermaid code. Do not include explanations, greetings, or any additional text.
                - Coordinates must be expressed as [Emotional Intensity Value, Formality Value] (not as tuples or in any other format).
                - Use the example format strictly, including title, axis labels, quadrant labels, and proper data representation.
                - Emotional Intensity level and Formality level should be converted as follows:
                    * If the level is 1, convert to 0 value.
                    * If the level is 2, convert to 0.25 value.
                    * If the level is 3, convert to 0.5 value.
                    * If the level is 4, convert to 0.75 value.
                    * If the level is 5, convert to 1 value.
                    * If the level is 0~1, the value equal level minus 1 and divided by 4.
                    * ex: agacement: 情感強度:2 正式性:3
                        - agacement: [0.25, 0.5]
                - Use the output template to response."""),
            ('system', """
                Additional Rules:
                - All possitive values which are larger than 0 and lower than 1.
                - Depending on the words input from the user."""),
            ('system', """Here is an example of the expected input:
                colère: 情感強度：3 正式性：3
                替代詞：
                1. mécontentement: 情感強度：1 正式性：2
                2. agacement: 情感強度：2 正式性：3
                3. indignation: 情感強度：3 正式性：4
                4. courroux: 情感強度：4 正式性：5
                5. rage: 情感強度：5 正式性：1"""),
            ('system', """Here is an example of the expected output:
                quadrantChart
                    title Emotional Intensity and Formality
                    x-axis Low Emotional Intensity --> High Emotional Intensity
                    y-axis Low Formality --> High Formality
                    colere: [0.5, 0.5]
                    mecontentement: [0, 0.25]
                    agacement: [0.25, 0.5]
                    indignation: [0.5, 0.75]
                    courroux: [0.75, 1]
                    rage: [1, 0]"""),
            ('system', """
                <root> ::= <quadrant_chart>
                <quadrant_chart> ::= "quadrantChart" <title> <axes> <data_points>
                <title> ::= "title Emotional Intensity and Formality"
                <axes> ::= "x-axis Low Emotional Intensity --> High Emotional Intensity" 
                        | "y-axis Low Formality --> High Formality"
                <data_points> ::= <data_point> | <data_point> <data_points>
                <data_point> ::= <word>: "[" <emotional_intensity_value> ", " <formality_value> "]"
                <word> ::= /* 任意單詞 */
                <emotional_intensity_value> ::= "0" | "0.25" | "0.5" | "0.75" | "1" | 0<=任何數值<=1
                <formality_value> ::= "0" | "0.25" | "0.5" | "0.75" | "1" | 0<=任何數值<=1
            """),
            ('user', 'Please use every words strictly to generate mermaid code of quadrantChart with the template:{input}'),
        ])
        FAISS_vectordb = "faiss_FrenchWords202"
        vectordb = FAISS.load_local(
            FAISS_vectordb,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Finish FAISS Load")
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        document_chain = create_stuff_documents_chain(llm, text_prompt)
        self.document_diagram_chain = code_prompt | llm
        self.document_emotion_chain = emotion_prompt | llm
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    def text_request(self ,user_input :str):
        RAG_context = []
        first_response = self.retrieval_chain.invoke({
            'input': user_input,
            'context': RAG_context
        })['answer']
        print(f"Finish Text Response: {first_response}")
        second_response = self.document_emotion_chain.invoke({
            'input': first_response
        })
        print(f"Finish Emotional Response: {second_response}")
        return first_response, second_response
    
    def mermaid_code_request(self, llm_text):
        code_response = self.document_diagram_chain.invoke({'input': f"{llm_text}"})
        print(f"Finish Code Request")
        return code_response

    def mermaid_diagram_request(self, llm_code):
        screenshot_bytes = generate_mermaid_quadrantChart(llm_code)
        # 利用 BytesIO 將圖片數據轉為可在 Gradio 中顯示的格式
        mermaid_diagram = Image.open(BytesIO(screenshot_bytes))
        mermaid_diagram.seek(0)  # 重設讀取指標
        print(f"Finish Mermaid Diagram")
        return mermaid_diagram
    
    def clean_mermaid_code(self, output_code):
        init = "%%{init: {'theme': 'default', 'themeVariables': {'quadrant1Fill': '#FFDDC1', 'quadrant2Fill': '#C1E1FF', 'quadrant3Fill': '#C1FFC1', 'quadrant4Fill': '#FFC1C1'}}}%%\n"
        if output_code.startswith("```mermaid"):
            output_code = output_code[len("```mermaid"):].strip()
        if output_code.endswith("```"):
            output_code = output_code[:-len("```")].strip()
        output_code = unidecode(output_code)
        return init + output_code

    def question_request(self, user_input):
        text_response, emotion_response = self.text_request(user_input)
        mermaid_code = self.mermaid_code_request(emotion_response)
        mermaid_code = self.clean_mermaid_code(mermaid_code)
        print(f"clean-code = \n{mermaid_code}")
        mermaid_diagram = self.mermaid_diagram_request(mermaid_code)
        print(f"Finish Answer")
        return text_response, emotion_response, mermaid_code, mermaid_diagram


class GradioUI():
    # 建立 Gradio 接口
    def __init__(self):
        user_input = []
        llm = LLM()
        with gr.Blocks() as app:
            with gr.Row():
                user_input = gr.Textbox(label="User Input", lines=2, placeholder="Enter your question here......")
                user_output_text_response = gr.Textbox(label="LLM Text Response", lines=5)
                user_output_emotion_response = gr.Textbox(label="LLM Emotion Response", lines=5)
            user_button = gr.Button("Send Question")
            with gr.Row():
                user_output_code_response = gr.Textbox(label="LLM Emotion Response (editable)", lines=5, interactive=True)
                mermaid_output = gr.Image(type="pil", label="Mermaid Diagram")
            mermaid_button = gr.Button("Re Mermaid Diagram")
            user_button.click(
                fn=llm.question_request, 
                inputs=user_input, 
                outputs=[user_output_text_response, user_output_emotion_response, user_output_code_response, mermaid_output]
            )
            mermaid_button.click(
                fn=llm.mermaid_diagram_request,
                inputs=user_output_code_response,
                outputs=mermaid_output
            )
        self.ControllerView = app
    def launch(self):
        self.ControllerView.launch(share = False, server_port=8080)

if __name__ == "__main__":
    webUI = GradioUI()
    webUI.launch()
