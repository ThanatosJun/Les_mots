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
            ('user', 'based on the context below :\n\n{context}\n to answer the question:{input}'),
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
            ('user', "seperate from 1 to 5 in intensity and formality range:{input}")
        ])

        code_prompt = ChatPromptTemplate.from_messages([
            ('system', """Generate mermaid code of quadrantChart diagram based on the following replacement words (替代詞).
                Instruction:
                - Respond **only** with valid Mermaid code. Do not include explanations, greetings, or any additional text.
                - Coordinates must be expressed as [Emotional Intensity Value, Formality Value] (not as tuples or in any other format).
                - Use the example format strictly, including title, axis labels, quadrant labels, and proper data representation.
                - Emontional intensity value and formality value is in float only."""),
            ('system', """Here is an example of the expected input:
                colère: 情感強度：3 正式性：3
                替代詞：
                1. mécontentement: 情感強度：1 正式性：2
                2. agacement: 情感強度：2 正式性：3
                3. indignation: 情感強度：3 正式性：4
                4. courroux: 情感強度：4 正式性：5
                5. rage: 情感強度：5 正式性：1"""),
                    # quadrant-1 HE HF
                    # quadrant-2 LE HF
                    # quadrant-3 LE LF
                    # quadrant-4 HE LF
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
                Additional Rules:
                - All possitive values.
                - Ensure all values are be between 0 and 1 in float.
                - The diagram should use the input data to compute coordinates as follows:
                    - Emotional Intensity Value = (情緒強度 minus 1) divided by 4.
                    - Formality Value = (正式性 minus 1) divided by 4.
                    - Word: [Emotional Intensity Value, Formality Value].
                        ex: colère: 情感強度：3 正式性：3
                            colere: [0.5, 0.5] (in mermaid code)
                            mécontentement: 情感強度：1 正式性：2
                            mecontentement: [0, 0.25] (in mermaid code)
                - Depending on the words input from the user."""),
            ('user', 'Please use following imformation strictly to generate mermaid code of quadrantChart with the template:{input}'),
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
        second_response = self.document_emotion_chain.invoke({
            'input': first_response
        })
        text_response = first_response + "\n" + second_response
        print(f"Finish Text Request\ntext_response = {text_response}")
        return text_response, second_response, first_response
    
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
        text_response, emotion_response, first_response = self.text_request(user_input)
        mermaid_code = self.mermaid_code_request(emotion_response)
        print(f"mermaid_code = {mermaid_code}")
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
