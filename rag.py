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

# ─────────────────────────────────────────────
# ÉTAPE 5 — QA Chain + fonction finale
# ─────────────────────────────────────────────

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """Tu es un assistant pédagogique spécialisé dans les notes de cours.
Réponds à la question uniquement à partir du contexte fourni.
Si la réponse n'est pas dans le contexte, dis clairement que tu ne sais pas.
Réponds toujours en français, de façon claire et structurée.

Contexte :
{context}

Question : {question}

Réponse :"""

    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def rag_answer(question: str) -> str:
    vectorstore = get_vectorstore()
    chain = build_qa_chain(vectorstore)
    return chain.invoke(question)

if __name__ == "__main__":
    res = rag_answer("Qu'est-ce que le RAG ?")
    print("\n=== RÉPONSE ===")
    print(res)