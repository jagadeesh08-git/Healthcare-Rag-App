import streamlit as st
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

st.title("🩺 Healthcare RAG Chatbot")

@st.cache_resource
def load_db():
    docs = []
    docs.extend(TextLoader("healthcare_policy.txt").load())
    docs.extend(TextLoader("healthcare_guidelines.txt").load())
    docs.extend(CSVLoader("healthcare_protocols.csv").load())


    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L3-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

pipe = pipeline("text2text-generation", model="t5-small")


def ask_ai(context, question):
    prompt = f"""
Context:
{context}

Question: {question}

Answer in 2 short lines and 4 bullet points. 
If not found, say: Information not found in documents.
"""
    result = pipe(prompt, max_new_tokens=150)[0]["generated_text"]
    return result


# ✅ INPUT BOX
question = st.text_input("Ask healthcare question:")

# ✅ RUN WHEN USER TYPES
if question:
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_ai(context, question)

    st.write("### 🧠 Answer:")
    st.write(answer)




