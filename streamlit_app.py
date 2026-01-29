import streamlit as st
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

st.title("🩺 Healthcare RAG Chatbot")

# LOAD DATABASE
@st.cache_resource
def load_db():
    docs = []
    docs.extend(TextLoader("healthcare_policy.txt").load())
    docs.extend(TextLoader("healthcare_guidelines.txt").load())
    docs.extend(CSVLoader("healthcare_protocols.csv").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # ✅ Fake embeddings (works on Streamlit Cloud)
    embeddings = FakeEmbeddings(size=384)

    db = Chroma.from_documents(chunks, embeddings)
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 3})


# SIMPLE AI ANSWER FUNCTION (NO HEAVY MODEL)
def ask_ai(context, question):
    if context.strip() == "":
        return "Information not found in documents."

    return f"""Explanation:
{context[:200]}

• Follow doctor advice  
• Maintain healthy diet  
• Exercise and yoga  
• Regular health checkup  
"""


# INPUT
question = st.text_input("Ask healthcare question:")

# RUN
if question:
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_ai(context, question)

    st.write("### 🧠 Answer:")
    st.write(answer)
