import faiss
import glob
import ollama
import pandas as pd
csv_file = "KnowledgeBase\French_Dictionary_Change.csv"
KnowledgeBases = "KnowledgeBase"   
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=csv_file)
data = loader.load()
print(data[0].page_content)
print(data[0].metadata["source"])
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
documents = text_splitter.split_documents(data[:100])  # 將文件分割成更小的部分
# print(documents)
from langchain_community.embeddings import OllamaEmbeddings
# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="llama3.2")
from langchain_community.vectorstores import FAISS
# 使用FAISS建立向量資料庫
vectordb = FAISS.from_documents(documents, embeddings)
# 將向量資料庫設為檢索器
retriever = vectordb.as_retriever()
from langchain_core.prompts import ChatPromptTemplate
# 設定提示模板，將系統和使用者的提示組合
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# 初始化Ollama模型
llm = OllamaLLM(model='llama3.2')
# 創建文件鏈，將llm和提示模板結合
document_chain = create_stuff_documents_chain(llm, prompt)

# 創建檢索鏈，將檢索器和文件鏈結合
retrieval_chain = create_retrieval_chain(retriever, document_chain)
context = []
# input_text = input('>>> ')
# while input_text.lower() != 'bye':
#     response = retrieval_chain.invoke({
#         'input': input_text,
#         'context': context
#     })
#     print(response['answer'])
#     context = response['context']
#     input_text = input('>>> ')
vectordb.save_local("faiss_index")