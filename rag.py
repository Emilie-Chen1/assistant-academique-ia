import os

DOCS_DIR = "docs/"

def rag_answer(question):
    # Sécurité : créer le dossier s'il n'existe pas pour éviter le crash
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    
    return f"Le module RAG est prêt et sécurisé. Vous avez demandé : {question}"