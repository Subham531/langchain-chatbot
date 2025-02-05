from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import faiss
import os

os.environ["OPENAI_API_KEY"] = "api-key"

with open("scraped_data.txt",'r',encoding='utf-8') as f:
    raw_data  = f.readlines()

data = [Document(page_content=text) for text in raw_data]

embeddings = OpenAIEmbeddings()


faiss_index = FAISS.from_documents(data,embeddings)

faiss_index.save_local('faiss_index')

print('Embeddings created and stored in FAISS vector store')

