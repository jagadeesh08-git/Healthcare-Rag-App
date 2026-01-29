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
retriever = db.as_retriever(search_kwargs={"k": 1})


# SIMPLE AI ANSWER FUNCTION (NO HEAVY MODEL)
def ask_ai(context, question):
    if context.strip() == "":
        return "Information not found in documents."

    explanation = context.split(".")[:3]  # take first 3 sentences
    explanation_text = ". ".join(explanation)

    return f"""
**Explanation:**  
{explanation_text}

• Take doctor prescribed medicine  
• Drink plenty of water and rest  
• Eat healthy food and do yoga  
• Visit a doctor if symptoms continue  
"""



# INPUT
question = st.text_input("Ask healthcare question:")

# RUN
if question:
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_ai(context, question)

    st.write("### 🧠 Answer:")
    st.markdown(answer)


