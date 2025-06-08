# src/simple_rag/load_rgph.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os

def load_all_rgph_reports(directory: str) -> List:
    """
    Charge tous les PDF RGPH du répertoire `directory`, en découpe chaque texte en chunks.
    Retourne une liste de documents (list[Document]).
    """
    docs = []
    # Itérez sur chaque fichier .pdf
    for filename in os.listdir(directory):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(directory, filename)
        loader = PyPDFLoader(path)
        # Charger le document entier
        full_doc = loader.load()
        # Fractionnement basé sur longueur pour RAG
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(full_doc)
        # Ajoute un métadonnée “source” pour savoir quel PDF d’origine
        for chunk in chunks:
            chunk.metadata["source_pdf"] = filename
        docs.extend(chunks)
    return docs
