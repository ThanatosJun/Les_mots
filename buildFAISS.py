from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
csv_file = "KnowledgeBase\French_Dictionary_Change.csv"
KnowledgeBases = "KnowledgeBase"   
loader = CSVLoader(file_path=csv_file)
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
documents = text_splitter.split_documents(data)
embeddings = OllamaEmbeddings(model="llama3.2")
vectordb = FAISS.from_documents(documents, embeddings)
vectordb.save_local("faiss_FrenchWords")