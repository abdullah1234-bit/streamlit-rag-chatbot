import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

app = FastAPI()
@app.get("/")
def root():
    return {"message": "FastAPI RAG bot is running! 🎉 Use /upload, /build-vectors, /ask"}


# ---- SETUP ----
embeddings_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
PROMPT = PromptTemplate(
    template="""
Human: Use the following context to answer the question with a concise and factual response (250 words max).
If the answer is unknown, say "I don't know."

<context>
{context}
</context>

Question: {question}

Assistant:
""",
    input_variables=["context", "question"]
)

def get_ollama_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

def load_and_split_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_store(docs):
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local("faiss_index")

def get_answer(query):
    db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})["result"]

# ---- API Routes ----

@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    os.makedirs("data", exist_ok=True)
    for file in files:
        with open(f"data/{file.filename}", "wb") as f:
            f.write(await file.read())
    return {"message": f"{len(files)} file(s) uploaded."}

@app.post("/build-vectors/")
def build_vector_store():
    if not os.path.exists("data") or not os.listdir("data"):
        return JSONResponse(content={"error": "No files uploaded."}, status_code=400)
    docs = load_and_split_documents()
    if not docs:
        return JSONResponse(content={"error": "No content extracted from PDFs."}, status_code=400)
    create_vector_store(docs)
    return {"message": "Vector store created."}

@app.get("/ask/")
def ask_question(query: str = Form(...)):
    try:
        answer = get_answer(query)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)