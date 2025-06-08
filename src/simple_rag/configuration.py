import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class RAGConfig:
    def __init__(self):
        # Charger la clé depuis .env
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # Choix du modèle d’embeddings
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # Choix du vectorstore. Ici FAISS local.
        self.vectorstore = None  # sera instancié plus bas
        # Répertoire pour stocker l’index FAISS
        self.faiss_index_path = "faiss_index_rgph"
