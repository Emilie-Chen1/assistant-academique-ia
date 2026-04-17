import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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


# ─────────────────────────────────────────────
# ÉTAPE 2 — Splitting en chunks
# ─────────────────────────────────────────────


def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150, length_function=len
    )
    splits = text_splitter.split_documents(pages)
    print(f"Nombre de chunks : {len(splits)}")
    return splits


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
        VECTORSTORE_PATH, embedding, allow_dangerous_deserialization=True
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


# ─────────────────────────────────────────────
# MÉMOIRE — Reformulation de la question
# ─────────────────────────────────────────────

def reformulate_question(question: str, history: list) -> str:
    """
    Reformule la question en tenant compte de l'historique
    pour que le retriever comprenne le contexte.
    Ex: "Et lui ?" → "Qu'est-ce que le RAG ?"
    """
    if not history:
        return question

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    history_text = "\n".join(
        f"{'Utilisateur' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in history[-6:]  # 3 derniers échanges max
    )

    template = """Tu es un assistant qui reformule des questions.
Voici l'historique de la conversation :
{history}

Question actuelle : {question}

Reformule la question actuelle de façon autonome et complète,
en intégrant le contexte nécessaire de l'historique.
Si la question est déjà claire et autonome, retourne-la telle quelle.
Réponds uniquement avec la question reformulée, sans explication.

Question reformulée :"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    reformulated = chain.invoke({
        "history": history_text,
        "question": question
    })
    print(f"Question originale : {question}")
    print(f"Question reformulée : {reformulated}")
    return reformulated


# ─────────────────────────────────────────────
# ÉTAPE 5 — QA Chain + fonction finale
# ─────────────────────────────────────────────


def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """Tu es un assistant pédagogique spécialisé dans les notes de cours.
Réponds à la question uniquement à partir du contexte fourni.
Pour chaque information importante, tu DOIS citer la source entre parenthèses (ex: Source: nom_du_fichier.pdf).
Si la réponse n'est pas dans le contexte, dis clairement que tu ne sais pas.
Réponds toujours en français, de façon claire et structurée.

Contexte :
{context}

Question : {question}

Réponse avec citations :"""

    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    def format_docs(docs):
        formatted_parts = []
        for doc in docs:
            source_file = os.path.basename(doc.metadata.get("source", "Inconnu"))
            page_num = doc.metadata.get("page", 0) + 1
            content = f"--- SOURCE: {source_file} (Page {page_num}) ---\n{doc.page_content}"
            formatted_parts.append(content)
        return "\n\n".join(formatted_parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, history: list | None = None) -> str:
    vectorstore = get_vectorstore()
    
    # Reformulation de la question avec l'historique
    question_reformulee = reformulate_question(question, history or [])
    
    chain = build_qa_chain(vectorstore)
    return chain.invoke(question_reformulee)


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "Qu'est-ce que le RAG ?"},
        {"role": "assistant", "content": "Le RAG est une technique qui combine LLM et base documentaire."},
    ]
    res = rag_answer("Et quels sont ses avantages ?", history)
    print("\n=== RÉPONSE ===")
    print(res)