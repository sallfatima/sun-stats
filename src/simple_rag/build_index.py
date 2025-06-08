# src/simple_rag/build_index.py

import os
from src.simple_rag.configuration import RAGConfig
from src.simple_rag.load_rgph import load_all_rgph_reports
from langchain.vectorstores import FAISS

def main():
    # Charger la configuration (embeddings, chemins, etc.)
    config = RAGConfig()

    # Dossier où sont stockés vos PDF RGPH
    directory = "vos_rapports_rgph"
    print("🔄 Chargement des documents RGPH…")
    docs = load_all_rgph_reports(directory)

    # Si l’index FAISS n'existe pas déjà, on le crée
    if not os.path.exists(config.faiss_index_path):
        print("📦 Création de l’index FAISS…")
        vectorstore = FAISS.from_documents(docs, config.embedding_model)
        os.makedirs(config.faiss_index_path, exist_ok=True)
        vectorstore.save_local(config.faiss_index_path)
        print(f"🎯 Index FAISS sauvegardé dans {config.faiss_index_path}/")
    else:
        print(f"✅ L’index FAISS existe déjà dans {config.faiss_index_path}/")

if __name__ == "__main__":
    main()
