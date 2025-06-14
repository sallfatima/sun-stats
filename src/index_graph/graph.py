# src/index_graph/graph.py

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import asyncio

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from shared import retrieval
from langgraph.graph import StateGraph, START, END

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
LOG_PATH = Path("indexing_errors.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


def check_index_config(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Valide la configuration et le répertoire PDF.
    """
    configuration = IndexConfiguration.from_runnable_config(config)

    # Validation de la clé API
    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")

    if configuration.api_key != os.getenv("INDEX_API_KEY"):
        raise ValueError("Authentication failed: Invalid API key provided.")

    if configuration.retriever_provider != "pinecone":
        raise ValueError(
            "Only Pinecone is currently supported for document indexing."
        )

    # Validation du répertoire PDF
    if not state.pdf_root:
        raise ValueError("pdf_root doit être spécifié et non vide")

    pdf_root = Path(state.pdf_root).expanduser().resolve()
    if not pdf_root.exists():
        raise FileNotFoundError(f"Répertoire PDF introuvable : {pdf_root}")
    
    if not pdf_root.is_dir():
        raise ValueError(f"pdf_root doit être un répertoire : {pdf_root}")

    # Compter les PDFs disponibles
    pdf_files = list(pdf_root.glob("**/*.pdf"))
    logging.info(f"📁 Répertoire validé: {pdf_root}")
    logging.info(f"📄 {len(pdf_files)} fichiers PDF trouvés")

    return {"total_files_found": len(pdf_files)}

async def index_local_pdfs(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Indexe tous les PDFs du répertoire local spécifié.
    """
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    
    # Recherche récursive de tous les PDFs
    pdf_paths = list(pdf_root.glob("**/*.pdf"))
    
    stats = {
        "processed_files": [],
        "failed_files": [],
        "total_chunks": 0,
        "status": "Indexation en cours...",
        "total_files_found": len(pdf_paths)
    }

    if not pdf_paths:
        return {
            **stats,
            "status": "Aucun fichier PDF trouvé dans le répertoire"
        }

    logging.info(f"🚀 Début de l'indexation de {len(pdf_paths)} PDFs")
    now_str = datetime.utcnow().isoformat()
    
    # SIMPLIFIÉ: Un seul retriever pour le texte
    async with retrieval.make_retriever(config) as retriever:
        vectorstore = retriever.vectorstore
        
        # Configuration du text splitter pour les documents locaux
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_texts, all_metadatas, all_ids = [], [], []

        for idx, pdf_path in enumerate(pdf_paths, 1):
            try:
                logging.info(f"📖 ({idx}/{len(pdf_paths)}) Traitement: {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                pages = await asyncio.to_thread(loader.load_and_split)
                
                file_chunks = 0
                for page_doc in pages:
                    text = page_doc.page_content.strip()
                    if not text or len(text) < 50:  # Ignorer les pages quasi-vides
                        continue

                    # Découper le texte en chunks
                    chunks = text_splitter.split_text(text)
                    
                    page_num = page_doc.metadata.get("page", 0)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        # ID unique pour chaque chunk
                        vector_id = f"{pdf_path.stem}__p{page_num}__c{chunk_idx}"

                        all_texts.append(chunk_text)
                        all_metadatas.append({
                            "page_num": page_num,
                            "chunk_idx": chunk_idx,
                            "pdf_path": str(pdf_path),
                            "pdf_name": pdf_path.name,
                            "pdf_dir": str(pdf_path.parent),
                            "file_size": pdf_path.stat().st_size,
                            "indexed_at": now_str,
                            "type": "rgph_text",  # Spécifique aux rapports RGPH
                            "chunk_length": len(chunk_text)
                        })
                        all_ids.append(vector_id)
                        file_chunks += 1
                
                if file_chunks > 0:
                    stats["processed_files"].append(pdf_path.name)
                    logging.info(f"✅ {pdf_path.name}: {file_chunks} chunks extraits")
                else:
                    logging.warning(f"⚠️ {pdf_path.name}: aucun contenu exploitable")
                        
            except Exception as e:
                logging.error(f"❌ Erreur {pdf_path.name}: {e}")
                stats["failed_files"].append(pdf_path.name)
                continue

        if not all_texts:
            return {
                **stats,
                "status": "Aucun contenu textuel exploitable trouvé"
            }

        # Indexation par lots optimisée
        batch_size = 50
        successfully_indexed = 0
        
        logging.info(f"💾 Indexation de {len(all_texts)} chunks en {(len(all_texts)-1)//batch_size + 1} lots")

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            
            try:
                # Utiliser la méthode d'ajout de texte du vectorstore
                if hasattr(vectorstore, 'add_texts'):
                    # Méthode synchrone dans un thread
                    await asyncio.to_thread(
                        vectorstore.add_texts,
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                elif hasattr(vectorstore, 'aadd_texts'):
                    # Méthode asynchrone
                    await vectorstore.aadd_texts(
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                else:
                    logging.error("Vectorstore ne supporte ni add_texts ni aadd_texts")
                    continue
                
                successfully_indexed += len(batch_texts)
                batch_num = i//batch_size + 1
                total_batches = (len(all_texts)-1)//batch_size + 1
                logging.info(f"✅ Lot {batch_num}/{total_batches}: {len(batch_texts)} chunks indexés")
                
            except Exception as e:
                logging.error(f"❌ Erreur lot {i//batch_size + 1}: {e}")
                continue

        # Résultat final
        success_rate = len(stats["processed_files"]) / len(pdf_paths) * 100
        final_status = (
            f"Indexation terminée: {successfully_indexed} chunks depuis "
            f"{len(stats['processed_files'])}/{len(pdf_paths)} PDFs "
            f"({success_rate:.1f}% de succès)"
        )

        logging.info(f"🎉 {final_status}")
        
        return {
            **stats,
            "status": final_status,
            "total_chunks": successfully_indexed
        }
# --- Construction du StateGraph ---
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)
builder.add_node("check_index_config", check_index_config)
builder.add_node("index_local_pdfs", index_local_pdfs)

builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "index_local_pdfs")
builder.add_edge("index_local_pdfs", END)

# Compilation du graphe
graph = builder.compile()
graph.name = "LocalPDFIndexGraph"