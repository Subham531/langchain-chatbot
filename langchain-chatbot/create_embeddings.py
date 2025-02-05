from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import faiss
import os

os.environ["OPENAI_API_KEY"] = "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"

with open("scraped_data.txt",'r',encoding='utf-8') as f:
    raw_data  = f.readlines()

data = [Document(page_content=text) for text in raw_data]

embeddings = OpenAIEmbeddings()


faiss_index = FAISS.from_documents(data,embeddings)

faiss_index.save_local('faiss_index')

print('Embeddings created and stored in FAISS vector store')

