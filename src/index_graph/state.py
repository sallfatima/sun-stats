"""
Gestion de l'état d'indexation (textuelle et visuelle) selon SOLID.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass(kw_only=True)
class InputState:
    """État des entrées pour l'indexation locale de documents."""

    pdf_root: str = field(
        default="",
        metadata={"description": "Dossier racine des PDFs à indexer."}
    )

@dataclass(kw_only=True)
class IndexState(InputState):
    """
    Structure de l'état pour l'indexation de documents locaux,
    incluant le suivi des fichiers traités et des images indexées.
    """
    status: str = field(
        default="",
        metadata={"description": "Statut actuel de l'opération d'indexation."}
    )
    processed_text_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs traités en texte."}
    )
    failed_text_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont le traitement textuel a échoué."}
    )
    processed_image_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont les images ont été indexées."}
    )
    failed_image_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Liste des PDFs dont l'indexation visuelle a échoué."}
    )
    total_text_chunks: int = field(
        default=0,
        metadata={"description": "Nombre total de segments textuels indexés."}
    )
    total_text_files_found: int = field(
        default=0,
        metadata={"description": "Nombre total de fichiers PDF trouvés."}
    )
    total_images_indexed: int = field(
        default=0,
        metadata={"description": "Nombre total d'images indexées."}
    )
    last_indexed_text: datetime | None = field(
        default=None,
        metadata={"description": "Horodatage du dernier index textuel."}
    )
    last_indexed_image: datetime | None = field(
        default=None,
        metadata={"description": "Horodatage du dernier index visuel."}
    )

    def mark_text_processed(self, pdf_path: str) -> None:
        if pdf_path not in self.processed_text_files:
            self.processed_text_files.append(pdf_path)
            self.last_indexed_text = datetime.utcnow()

    def mark_text_failed(self, pdf_path: str) -> None:
        if pdf_path not in self.failed_text_files:
            self.failed_text_files.append(pdf_path)

    def mark_image_processed(self, pdf_path: str) -> None:
        if pdf_path not in self.processed_image_files:
            self.processed_image_files.append(pdf_path)
            self.last_indexed_image = datetime.utcnow()

    def mark_image_failed(self, pdf_path: str) -> None:
        if pdf_path not in self.failed_image_files:
            self.failed_image_files.append(pdf_path)

    def has_text_processed(self, pdf_path: str) -> bool:
        return pdf_path in self.processed_text_files

    def has_image_processed(self, pdf_path: str) -> bool:
        return pdf_path in self.processed_image_files