# src/index_graph/graph.py
"""
Pipeline d'indexation locale : texte et images.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime
import asyncio

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from index_graph.image_indexer import ImageIndexer
from shared import retrieval
from langgraph.graph import StateGraph, START, END

# Charger env
load_dotenv()

# Setup logging
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
    Valide config et existence du dossier PDF.
    """
    configuration = IndexConfiguration.from_runnable_config(config)

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")
    if configuration.api_key != os.getenv("INDEX_API_KEY"):
        raise ValueError("Authentication failed: Invalid API key provided.")
    if configuration.retriever_provider != "pinecone":
        raise ValueError(
            "Only Pinecone is currently supported for document indexing."
        )

    if not state.pdf_root:
        raise ValueError("pdf_root must be specified and non-empty")
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    if not pdf_root.exists() or not pdf_root.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_root}")

    pdf_files = list(pdf_root.glob("**/*.pdf"))
    return {"total_files_found": len(pdf_files)}


async def index_local_pdfs(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Indexation des PDF et des images extraites.
    """
    cfg = IndexConfiguration.from_runnable_config(config)
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    pdf_paths = list(pdf_root.glob("**/*.pdf"))

    stats = {
        "processed_files": [],
        "failed_files": [],
        "total_chunks": 0,
        "status": "Indexation en cours...",
        "total_files_found": len(pdf_paths)
    }

    if not pdf_paths:
        stats["status"] = "Aucun fichier PDF trouvé"
        return stats

    # Prépare retriever textuel
    async with retrieval.make_retriever(config) as retriever:
        vectorstore = retriever.vectorstore
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialiser image indexer
        image_indexer = ImageIndexer(
            index_path=cfg.chart_index_path,
            images_dir=cfg.images_dir,
            pinecone_api_key=cfg.pinecone_api_key,
            pinecone_env=cfg.pinecone_env,
            openai_api_key=cfg.api_key,
            index_name=cfg.pinecone_image_index,
            embedding_model=cfg.vision_embedding_model
        )

        for idx, pdf in enumerate(pdf_paths, 1):
            try:
                # 1) Texte
                loader = PyPDFLoader(str(pdf))
                pages = await asyncio.to_thread(loader.load_and_split)
                texts, metas, ids = [], [], []
                now = datetime.utcnow().isoformat()
                for page_doc in pages:
                    content = page_doc.page_content.strip()
                    if len(content) < 50:
                        continue
                    chunks = splitter.split_text(content)
                    page_num = page_doc.metadata.get("page", 0)
                    for ci, txt in enumerate(chunks):
                        vid = f"{pdf.stem}__p{page_num}__c{ci}"
                        texts.append(txt)
                        metas.append({
                            **page_doc.metadata,
                            "pdf_path": str(pdf),
                            "indexed_at": now,
                            "type": "rgph_text"
                        })
                        ids.append(vid)
                # index text batch
                await asyncio.to_thread(
                    vectorstore.add_texts,
                    texts=texts,
                    metadatas=metas,
                    ids=ids
                )
                state.mark_text_processed(str(pdf))
                stats["processed_files"].append(pdf.name)

                # 2) Images
                image_indexer.index_all()
                state.mark_image_processed(str(pdf))

            except Exception as e:
                logging.error(f"Error processing {pdf.name}: {e}")
                stats["failed_files"].append(pdf.name)

        stats["total_chunks"] = len(stats["processed_files"])  # approx
        stats["status"] = "Indexation terminée"
        return stats

# Construction du graphe
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)
builder.add_node("check_index_config", check_index_config)
builder.add_node("index_local_pdfs", index_local_pdfs)
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "index_local_pdfs")
builder.add_edge("index_local_pdfs", END)

graph = builder.compile()
graph.name = "LocalPDFIndexGraph"
