from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
csv_file = "KnowledgeBase/French_Dictionary_Change.csv"
loader = CSVLoader(file_path=csv_file)
print (loader)
data = loader.load()
print("data load finished")
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
documents = text_splitter.split_documents(data)
print("text spliter finished")
embeddings = OllamaEmbeddings(model="llama3.2")
vectordb = FAISS.from_documents(documents, embeddings)
print("FAISS vector finished")
vectordb.save_local("faiss_FrenchWords")
print("FAISS Save in faiss_FrenchWords")