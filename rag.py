import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Chargement de la clé API
load_dotenv(find_dotenv())

DOCS_DIR = "docs/"
VECTORSTORE_PATH = "docs/faiss_index"

# ─────────────────────────────────────────────
# ÉTAPE 1 — Chargement des PDF
# ─────────────────────────────────────────────

def load_documents():
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    print(f"PDF trouvés : {pdf_files}")
    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCS_DIR, pdf_file))
        pages.extend(loader.load())
    print(f"Total pages chargées : {len(pages)}")
    return pages

if __name__ == "__main__":
    pages = load_documents()

# ─────────────────────────────────────────────
# ÉTAPE 2 — Splitting en chunks
# ─────────────────────────────────────────────

def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    splits = text_splitter.split_documents(pages)
    print(f"Nombre de chunks : {len(splits)}")
    return splits

if __name__ == "__main__":
    pages = load_documents()
    splits = split_documents(pages) 

# ─────────────────────────────────────────────
# ÉTAPES 3 & 4 — Vectorstore FAISS + Retrieval
# ─────────────────────────────────────────────

def build_vectorstore(splits):
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embedding)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("Vectorstore créé et sauvegardé")
    return vectorstore

def load_vectorstore():
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )
    print("Vectorstore chargé depuis le disque")
    return vectorstore

def get_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        return load_vectorstore()
    else:
        pages = load_documents()
        splits = split_documents(pages)
        return build_vectorstore(splits)

if __name__ == "__main__":
    pages = load_documents()
    splits = split_documents(pages)
    vectorstore = get_vectorstore()