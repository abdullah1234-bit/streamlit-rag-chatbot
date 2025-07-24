import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ---- Embeddings & LLM Setup ----
embeddings_model = OllamaEmbeddings(model="deepseek-r1:1.5b")

def get_ollama_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

# ---- PDF Ingestion ----
def load_and_split_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ---- Vector Store ----
def create_vector_store(docs):
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local("faiss_index")

# ---- Prompt ----
prompt_template = """
Human: Use the following context to answer the question with a concise and factual response (250 words max).
If the answer is unknown, say "I don't know."

<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---- QA Function ----
def get_answer(query):
    db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})['result']

# ---- Streamlit App ----
def main():
    st.set_page_config("Chat with PDF Locally")
    st.header("📄 Chat with PDF using Ollama 💬")

    # --- Upload PDFs ---
    uploaded_files = st.file_uploader("Upload PDF files:", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join("data", f.name), "wb") as out_file:
                out_file.write(f.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    # --- Sidebar: Vector Creation ---
    with st.sidebar:
        if st.button("Create/Update Vector Store"):
            with st.spinner("Processing documents..."):
                docs = load_and_split_documents()
                if docs:  # ✅ Only create the vector store if documents exist
                    create_vector_store(docs)
                    st.success("Vector store updated successfully!")
                else:
                    st.warning("No PDF files found in the data folder. Please upload files first.")


    # --- Ask a Question ---
    query = st.text_input("Ask a question:")
    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.write(answer)
            st.success("Done!")

if __name__ == "__main__":
    main()
