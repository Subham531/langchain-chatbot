from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import conversational_retrieval
from langchain_community.llms import openai
import os

os.environ["OPENAI_API_KEY"] = "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"

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36"


url = "https://brainlox.com/courses/category/technical"

loader = WebBaseLoader(url)

data = loader.load()

for document in data:
    print(document.page_content)


with open("scraped_data.txt","w",encoding='utf-8') as f:
    for document in data:
        f.write(document.page_content +'\n')

