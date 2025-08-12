# server.py
# This script runs a local web server to handle AI classification and chat.

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import pandas as pd
import os

# --- INITIAL SETUP (Runs once when the server starts) ---
print("Starting server and setting up AI model...")
load_dotenv()

# Check for Google API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it up in a .env file.")

# a. Document Ingestion
try:
    df = pd.read_csv("email_spam.csv")
    df['content'] = df['title'] + "\n" + df['text']
    documents = df['content'].tolist()
    labels = df['type'].tolist()
except FileNotFoundError:
    raise FileNotFoundError("Error: email_spam.csv not found. Make sure it's in the same directory as server.py.")

# b. Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for doc, label in zip(documents, labels):
    chunks = splitter.split_text(doc)
    for chunk in chunks:
        doc_obj = Document(page_content=chunk, metadata={"label": label})
        docs.append(doc_obj)

# c. Embedding Generation and Storing
print("Generating embeddings and creating vector store...")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_documents(docs, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store is ready.")

# d. Model and Chains Setup
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# --- Chain 1: Initial Spam Classification ---
clf_prompt = PromptTemplate(
    template="""You are an email spam classifier. Based on the context of similar emails, classify the given email.
    Answer ONLY with the single word "Spam" or "Not Spam". Do not provide any explanation.

    Context:
    {context}

    Given email:
    {email}""",
    input_variables=['context', 'email']
)

def format_docs(retrieved_docs):
  return "\n\n".join(doc.page_content for doc in retrieved_docs)

clf_chain = (
    {"context": retriever | format_docs, "email": RunnablePassthrough()}
    | clf_prompt
    | model
    | StrOutputParser()
)

# --- Chain 2: Conversational Chat ---
chat_prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant for analyzing emails. Answer the user's question based on the provided email content and chat history.
    Keep your answers concise and conversational.

    Chat History:
    {chat_history}

    Email Content:
    {email_content}

    User Question:
    {question}"""
)

def format_chat_history(history):
    """Formats the chat history list of objects into a readable string."""
    if not history:
        return "No history yet."
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])


chat_chain = (
    {
        "question": lambda x: x["question"],
        "email_content": lambda x: x["email_content"],
        "chat_history": lambda x: format_chat_history(x["chat_history"]),
    }
    | chat_prompt
    | model
    | StrOutputParser()
)

print("AI Model and chains are ready.")
# --- END OF INITIAL SETUP ---


# --- FLASK APP ---
app = Flask(__name__)
# CORS allows the extension (from a different "origin") to communicate with this server.
CORS(app)

@app.route('/classify', methods=['POST'])
def classify_email():
    """Endpoint for the initial spam classification."""
    data = request.json
    email_text = data.get('email')
    print(email_text)
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    print("Classifying email...")
    result = clf_chain.invoke(email_text)
    print(f"Classification result: {result}")
    return jsonify({"classification": result.strip()})

@app.route('/chat', methods=['POST'])
def chat_with_email():
    """Endpoint for follow-up questions about the email."""
    data = request.json
    question = data.get('question')
    email_content = data.get('email_content')
    chat_history = data.get('chat_history', []) # Expects a list of message objects

    if not all([question, email_content]):
        return jsonify({"error": "Missing 'question' or 'email_content'"}), 400

    print(f"Handling chat query: {question}")
    result = chat_chain.invoke({
        "question": question,
        "email_content": email_content,
        "chat_history": chat_history
    })
    print(f"Chat response: {result}")
    return jsonify({"response": result.strip()})

if __name__ == '__main__':
    print("Server is running at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
