import gradio as gr
from mermaid import generate_mermaid_quadrantChart
from PIL import Image
from io import BytesIO
from unidecode import unidecode
import ollama

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
grammar_path = "llama.cpp/grammars/japanese.gbnf"
llm = OllamaLLM(model='llama3.2', grammar_path = grammar_path,  temperature=0.7)
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
FAISS_vectordb = "faiss_FrenchWords202"
vectordb = FAISS.load_local(
    FAISS_vectordb,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
document_chain = create_stuff_documents_chain(llm, text_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

document_emotion_chain = emotion_prompt | llm
# first_response = """
# 1. Bonsoir：表示晚上的問候，通常在日落後的時間使用。
# 2. Salut：一種口語問候，簡單且 friendly，可以用於日常生活中的交際。
# 3. Bonjour le matin/le soir：表示早上或下午的問候，類似於 Bonsoir，但時間稍有不同。
# 4. Bonne journée/Bonne nuit：表示一個好的日子 or 晚，簡單但表達了友善的意念。
# 5. Bonheur：是一種formal 的問候，表示和他人交際時的溫暖和親切。"""

user_input = "請給我5個法文bonjour的同義詞"

code_prompt = ChatPromptTemplate.from_messages([
    ('system', """Generate mermaid code of quadrantChart diagram based on the following replacement words (替代詞).
        Instruction:
        - Respond **only** with valid Mermaid code. Do not include explanations, greetings, or any additional text.
        - Coordinates must be expressed as [Emotional Intensity Value, Formality Value] (not as tuples or in any other format).
        - Use the example format strictly, including title, axis labels, quadrant labels, and proper data representation.
        - Emontional intensity value and formality value is in float only."""),
    ('system', """
        Additional Rules:
        - All possitive values.
        - Ensure all values are lower than 1 and larger than 0 in float.
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
document_diagram_chain = code_prompt | llm

first_response = retrieval_chain.invoke({
    'input': user_input,
    'context': []
})['answer']
print(first_response)
second_response = document_emotion_chain.invoke({
    'input': first_response
})
print(second_response)
code_response = document_diagram_chain.invoke({'input': f"{second_response}"})
print(code_response)