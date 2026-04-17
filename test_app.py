import pytest
from unittest.mock import MagicMock, patch

# Import des fonctions existantes
from rag import rag_answer
from agents import agent_answer
from app import auto_route

# 1. TEST DU ROUTAGE
@patch('langchain_openai.ChatOpenAI.invoke')
def test_auto_route_logic(mock_invoke):
    """Vérifie que le routeur renvoie la bonne catégorie."""
    mock_invoke.return_value.content = "rag"
    assert auto_route("Explique moi le cours") == "rag"

# 2. TEST DU MODULE RAG
@patch('rag.get_vectorstore')
@patch('rag.build_qa_chain')
def test_rag_answer_mocked(mock_build_chain, mock_vectorstore):
    """
    Vérifie le RAG en mockant le retour final de la chaîne.
    Cette méthode évite les erreurs de validation Pydantic.
    """
    # Simulation du Vectorstore
    mock_vectorstore.return_value = MagicMock()
    
    # Simulation de la chaîne de QA
    mock_chain = MagicMock()
    # On force invoke() à renvoyer une chaîne de caractères pure
    mock_chain.invoke.return_value = "D'après vos cours, le RAG est utile."
    mock_build_chain.return_value = mock_chain

    # Exécution
    response = rag_answer("C'est quoi le RAG ?")
    
    assert isinstance(response, str)
    assert "D'après vos cours" in response

# 3. TEST DU MODULE AGENT
@patch('agents._agent.invoke')
def test_agent_answer_mocked(mock_agent_invoke):
    """Vérifie que l'agent renvoie une réponse structurée (dict)."""
    # Simulation du retour de l'agent
    mock_agent_invoke.return_value = {
        "messages": [
            MagicMock(content="Le résultat est 4")
        ]
    }

    result = agent_answer("Combien font 2+2 ?")
    
    assert isinstance(result, dict)
    assert result["source"] == "agent"
    assert "4" in str(result["content"])