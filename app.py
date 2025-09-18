from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.prompt import *
from src.helpers import download_hugging_face_embeddings
from store_index import docsearch

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOllama(model="mistral",base_url="http://localhost:11434")

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", system_prompt),
    ("human", "{input}")
  ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print("User Input:", input)
    response = rag_chain.invoke(
      {"input": input}
    )
    print("Chatbot Response:", response['answer'])
    return str(response['answer'])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)