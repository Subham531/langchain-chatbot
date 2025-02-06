from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

app = Flask(__name__)


os.environ["OPENAI_API_KEY"] = "api-key"
embeddings = OpenAIEmbeddings()


faiss_index = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # It searches for relevant documents in the FAISS index
    docs = faiss_index.similarity_search(user_input, k=5)  # k=5 is the number of results to retrieve
    
    #  It generates the chatbot response
    response = generate_response(docs, user_input)
    
    return jsonify({"response": response})

def generate_response(docs, user_input):
   
    context = "\n".join([doc.page_content for doc in docs]) 
    prompt = f"User Input: {user_input}\n\nContext: {context}\n\nResponse:"
    
   
    response = OpenAIEmbeddings().query(prompt)
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
