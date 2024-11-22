import chromadb
import pandas as pd
import ollama

Basic_prompt = """請將

"""
def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")  # 生成用戶輸入的嵌入向量
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)  # 在集合中查詢最相關的三個文檔
    data = results['documents'][0]  # 獲取最相關的文檔
    output = ollama.generate(
        model="llama3.2",
        prompt=f"Using this data: {data}. Respond to this prompt and use zh-TW: {user_input}"  # 生成回應
    )
    print(output)
    return output

def initial():
    # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
    client = chromadb.PersistentClient(path="demodocs")
    file_path = 'QA50.csv'  # 指定Excel文件的路徑和名稱
    documents = pd.read_csv(file_path, header=None)  # 使用pandas讀取Excel文件
    # 使用chromadb客戶端創建或獲取名為'demodocs'的集合
    collection = client.get_or_create_collection(name="demodocs")
    print(f"Knowledge data Ready!")
    for index, content in documents.iterrows():
        print(f"index = {index}")
        print(f"content = {content[0]}")
        print(f"collection = {collection}")
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content[0])  # 通過ollama生成該行文本的嵌入向量
        collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]])  # 將文本和其嵌入向量添加到集合中
    print(f"Knowledge data finished turn into vector!")
    return collection

collection = initial()
user_input = ""
user_input = input("請輸入您的問題：")
while(user_input != "#END"):
    output = handle_user_input(user_input, collection)
    print(output["response"])
    user_input = input("請輸入您的問題：")


