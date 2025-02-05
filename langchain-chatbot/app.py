from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

app = Flask(__name__)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()

# Load the FAISS index with embeddings, allowing dangerous deserialization
faiss_index = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Search for relevant documents in the FAISS index
    docs = faiss_index.similarity_search(user_input, k=5)  # k=5 is the number of results to retrieve
    
    # Generate the chatbot response
    response = generate_response(docs, user_input)
    
    return jsonify({"response": response})

def generate_response(docs, user_input):
    # Here you can customize the logic to process documents and input
    # Use the OpenAI model to generate a response based on the documents and user input
    context = "\n".join([doc.page_content for doc in docs])  # Combine the relevant documents as context
    prompt = f"User Input: {user_input}\n\nContext: {context}\n\nResponse:"
    
    # Get OpenAI model response (You can replace this with your own logic to send to OpenAI API)
    response = OpenAIEmbeddings().query(prompt)
    
    return response

if __name__ == '__main__':
    app.run(debug=True)