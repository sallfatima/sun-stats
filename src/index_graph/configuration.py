# index_graph/configuration.py
from __future__ import annotations
from dataclasses import dataclass, field
from shared.configuration import BaseConfiguration
from dotenv import load_dotenv
import os

load_dotenv()  # charge .env à la racine du projet

DEFAULT_DOCS_FILE = "src/sample_docs.json"

@dataclass(kw_only=True)
class IndexConfiguration(BaseConfiguration):
    """Configuration pour l’indexation textuelle et visuelle."""

    # — Textuel (existant) —
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        metadata={"description": "Clé OpenAI pour embeddings textuels."}
    )
    pinecone_index: str = field(
        default="ansd-doc",
        metadata={"description": "Nom de l’index Pinecone pour le texte."}
    )

    # — Visuel (nouveau) —
    chart_index_path: str = field(
        default_factory=lambda: os.getenv("CHART_INDEX_PATH", "charts_index.csv"),
        metadata={"description": "Chemin vers le CSV d’index des graphiques."}
    )
    images_dir: str = field(
        default_factory=lambda: os.getenv("IMAGES_DIR", "src/assets/images"),
        metadata={"description": "Dossier racine des images extraites."}
    )
    vision_embedding_model: str = field(
        default="clip-vision-512",
        metadata={"description": "Modèle OpenAI pour embeddings vision."}
    )
    text_embedding_model: str = field(
        default="text-embedding-ada-002",
        metadata={"description": "Modèle OpenAI pour embeddings texte."}
    )
    pinecone_image_index: str = field(
        default="sunu-visual-index",
        metadata={"description": "Nom de l’index Pinecone pour les images."}
    )
