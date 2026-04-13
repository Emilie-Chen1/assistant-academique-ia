from rag import rag_answer
from agents import agent_answer

def test_functions_exist():
    # On vérifie juste que les fonctions renvoient bien du texte
    assert isinstance(rag_answer("test"), str)
    assert isinstance(agent_answer("test"), str)