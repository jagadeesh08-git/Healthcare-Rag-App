from transformers import pipeline
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load documents
docs = []
docs.extend(TextLoader("data/healthcare_policy.txt").load())
docs.extend(TextLoader("data/healthcare_guidelines.txt").load())
docs.extend(CSVLoader("data/healthcare_protocols.csv").load())

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector DB
db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
retriever = db.as_retriever(search_kwargs={"k": 4})

# Local AI model
pipe = pipeline("text2text-generation", model="google/flan-t5-large")



# Better AI prompt
def ask_ai(context, question):
    prompt = f"""
You are a healthcare expert assistant.
Explain clearly for adults in exactly 4 short lines.
Use ONLY the provided context. Do not add extra knowledge.
If the answer is not in the context, say: "Information not found in documents."

Context:
{context}

Question:
{question}

Answer:
"""
    result = pipe(prompt, max_new_tokens=200)[0]["generated_text"]
    return result


# Chat loop
while True:
    q = input("Ask healthcare question (type exit to stop): ")
    if q.lower() == "exit":
        break

    relevant_docs = retriever.invoke(q)
    context = " ".join([d.page_content for d in relevant_docs])

    answer = ask_ai(context, q)
    print("\nAnswer:", answer)
