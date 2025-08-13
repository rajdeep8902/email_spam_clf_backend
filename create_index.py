# create_index.py
# This script builds the ChromaDB vector store and saves it to a folder.
# Run this file ONCE on your local machine.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma # Changed from FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
import os
import shutil

print("Starting ChromaDB index creation process...")
load_dotenv()

# Make sure your GOOGLE_API_KEY is in a .env file
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please set it up in a .env file before running.")

# Define the path for the ChromaDB database
CHROMA_PATH = "chroma_db"

# Clean up old database directory if it exists
if os.path.exists(CHROMA_PATH):
    print(f"Removing old database directory: {CHROMA_PATH}")
    shutil.rmtree(CHROMA_PATH)

# 1. Document Ingestion
try:
    df = pd.read_csv("email_spam.csv")
    df['content'] = df['title'] + "\n" + df['text']
    documents = df['content'].tolist()
    labels = df['type'].tolist()
    print("Successfully loaded email_spam.csv")
except FileNotFoundError:
    raise FileNotFoundError("Error: email_spam.csv not found. Make sure it's in the same directory.")

# 2. Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for doc, label in zip(documents, labels):
    chunks = splitter.split_text(doc)
    for chunk in chunks:
        doc_obj = Document(page_content=chunk, metadata={"label": label})
        docs.append(doc_obj)
print(f"Split documents into {len(docs)} chunks.")

# 3. Embedding Generation and Storing in Chroma
print("Generating embeddings and creating ChromaDB vector store. This may take a moment...")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Create the Chroma vector store and persist it to disk
vector_store = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)
print("Vector store created successfully.")


print(f"\nSUCCESS: The ChromaDB index has been created and saved to the '{CHROMA_PATH}' folder.")
print(f"You can now upload this '{CHROMA_PATH}' folder to your GitHub repository.")
