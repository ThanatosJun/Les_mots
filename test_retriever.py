from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
print(create_retrieval_chain)
FAISS_vectordb = "faiss_FrenchWords202"
llm = OllamaLLM(model='llama3.2')
embeddings = OllamaEmbeddings(model="llama3.2")
vectordb = FAISS.load_local(
    FAISS_vectordb,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
text_prompt = ChatPromptTemplate.from_messages([
    ('system', 
        """You will act as an expert in French, skilled at analyzing the semantics, grammatical properties, and contextual applications of French words. Based on a given word, you will identify synonyms or near-synonyms with the same grammatical category.
        Task Description:
        1. Your goal is to analyze a given French word, identify its synonyms (同義詞).
        2. You will identify 5 synonyms with the same grammatical category as the given word.
        3. Use zh-TW 繁體中文 to respond."""),
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
document_chain = create_stuff_documents_chain(llm, text_prompt)
print(f"Finish FAISS Load")
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print(retrieval_chain)  # 打印返回的對象，檢查其內容
print("FINISH")