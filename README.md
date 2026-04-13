# 🎓 Assistant Académique IA

Assistant intelligent conçu pour faciliter la recherche académique en utilisant le **RAG (Retrieval-Augmented Generation)** et des **Agents IA**.

## 📋 Prérequis
- Python : 3.10 (version recommandée pour la compatibilité des outils IA)
- Gestionnaire de paquets : pip


## 📁 Structure des fichiers
- `app.py` : Interface utilisateur Streamlit et logique de routage.
- `rag.py` : Module de récupération de documents (RAG).
- `agents.py` : Module des agents de recherche web.
- `requirements.txt` : Dépendances du projet (LangChain, OpenAI, etc.).
- `.github/workflows/ci.yml` : Pipeline d'intégration continue.

## 🛠️ Installation et Lancement local

1. **Clonage du dépôt** :
   ```bash
   git clone [https://github.com/Emilie-Chen1/assistant-academique-ia.git](https://github.com/Emilie-Chen1/assistant-academique-ia.git)
   cd assistant-academique-ia
    ```

2. **Installation des dépendances** : 
    ```bash
    pip install -r requirements.txt
    ```

3. **Configuration des API** : Créez un fichier .env à la racine du projet et ajoutez vos clés (ce fichier est ignoré par Git).
    ```bash
    GOOGLE_API_KEY=votre_cle_gemini
    TAVILY_API_KEY=votre_cle_tavily
    ```

4. **Lancement de l'application** :
    ```bash
    streamlit run app.py
    ```

## ✅ Standard de Qualité

Le projet utilise des outils automatisés pour garantir la propreté du code :

- Formatage : Black (longueur de ligne 88).
- Linting : Pylint et Flake8.
- Tests : Pytest.

Les tests sont déclenchés automatiquement sur GitHub à chaque modification du code.