# src/index_graph/graph.py
"""
Pipeline d'indexation locale complet : extraction + indexation de texte, images et tableaux.
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
from index_graph.pdf_visual_extractor import PDFVisualExtractor
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

    # if not configuration.api_key:
    #     raise ValueError("API key is required for document indexing.")
    # if configuration.api_key != os.getenv("INDEX_API_KEY"):
    #     raise ValueError("Authentication failed: Invalid API key provided.")
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


async def extract_visual_content(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Étape 1: Extraire les graphiques et tableaux des PDFs.
    """
    print("🔍 ÉTAPE 1: Extraction du contenu visuel des PDFs...")
    
    cfg = IndexConfiguration.from_runnable_config(config)
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    
    # Vérifier si l'extraction visuelle est activée
    if not cfg.enable_visual_indexing:
        print("⚠️ Extraction visuelle désactivée dans la configuration")
        # IMPORTANT: Retourner l'état avec visual_extraction_completed = True
        # même si désactivé pour permettre l'indexation textuelle
        return {
            "visual_extraction_completed": True,
            "total_images_extracted": 0,
            "total_tables_extracted": 0,
            "extraction_errors": ["Extraction visuelle désactivée dans la configuration"]
        }
    
    try:
        # Créer l'extracteur visuel
        extractor = PDFVisualExtractor(
            output_dir=cfg.output_dir if hasattr(cfg, 'output_dir') else ".",
            images_dir=cfg.images_dir,
            tables_dir=cfg.tables_dir
        )
        
        # Traiter tous les PDFs du dossier
        print(f"📁 Extraction depuis: {pdf_root}")
        
        # Exécuter l'extraction dans un thread séparé pour éviter le blocage
        extraction_result = await asyncio.to_thread(
            extractor.process_directory,
            pdf_root
        )
        
        print(f"✅ Extraction terminée:")
        print(f"   📊 {extraction_result['total_images']} images extraites")
        print(f"   📋 {extraction_result['total_tables']} tableaux extraits")
        print(f"   📄 {extraction_result['processed_pdfs']} PDFs traités")
        
        # IMPORTANT: Retourner les mises à jour de l'état au bon format pour LangGraph
        # Ces clés seront fusionnées avec l'état existant
        return {
            "visual_extraction_completed": True,
            "total_images_extracted": extraction_result["total_images"],
            "total_tables_extracted": extraction_result["total_tables"],
            "extraction_errors": extraction_result.get("processing_errors", [])
        }
        
    except Exception as e:
        error_msg = f"Erreur lors de l'extraction visuelle: {e}"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        
        # En cas d'erreur, marquer comme non complété mais permettre le reste
        return {
            "visual_extraction_completed": False,
            "total_images_extracted": 0,
            "total_tables_extracted": 0,
            "extraction_errors": [error_msg]
        }

async def extract_text_from_image_content(image_path: Path, api_key: str) -> str:
    """Extrait le texte d'une image via l'API Vision OpenAI."""
    import base64
    import openai
    
    if not image_path.exists():
        print(f"         ⚠️ Image non trouvée: {image_path}")
        return ""
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Configuration client OpenAI
        client = openai.OpenAI(api_key=api_key)
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extrayez tout le texte visible dans cette image ANSD (graphique/tableau). 
                        Incluez titres, légendes, valeurs numériques, labels d'axes, unités.
                        Organisez le texte de manière structurée et logique.
                        Si c'est un graphique, décrivez les données principales.
                        Répondez en français."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=1000,
            temperature=0
        )
        
        content = response.choices[0].message.content
        return content if content else ""
        
    except Exception as e:
        logging.error(f"Erreur extraction vision {image_path}: {e}")
        print(f"         ❌ Erreur API Vision: {e}")
        return ""



async def extract_text_from_table_content(table_path: Path) -> str:
    """Extrait le texte d'un fichier CSV de tableau."""
    import pandas as pd
    
    if not table_path.exists():
        return ""
    
    try:
        # Lire le CSV du tableau
        df = pd.read_csv(table_path)
        
        # Convertir en format texte structuré
        if df.empty:
            return ""
            
        # Limiter le nombre de lignes pour éviter les contenus trop volumineux
        max_rows = 20
        if len(df) > max_rows:
            df = df.head(max_rows)
            truncated_note = f"\n[Tableau tronqué: affichage de {max_rows} lignes sur {len(pd.read_csv(table_path))}]"
        else:
            truncated_note = ""
        
        # Formatage du tableau en texte
        text_parts = []
        
        # En-têtes
        headers = " | ".join(str(col) for col in df.columns)
        text_parts.append(f"Colonnes: {headers}")
        
        # Données (lignes principales)
        for idx, row in df.iterrows():
            if idx >= 5:  # Limiter aux 5 premières lignes pour l'indexation
                break
            row_text = " | ".join(str(val) for val in row.values)
            text_parts.append(f"Ligne {idx + 1}: {row_text}")
        
        # Statistiques du tableau
        text_parts.append(f"Dimensions: {len(df)} lignes × {len(df.columns)} colonnes")
        
        result = "\n".join(text_parts) + truncated_note
        return result
        
    except Exception as e:
        logging.error(f"Erreur lecture tableau {table_path}: {e}")
        return ""

async def index_local_pdfs(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Étape 2: Indexation complète - texte des PDFs + contenu visuel extrait.
    """
    print("📚 ÉTAPE 2: Indexation du contenu textuel et visuel...")
    
    cfg = IndexConfiguration.from_runnable_config(config)
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    pdf_paths = list(pdf_root.glob("**/*.pdf"))

    stats = {
        "processed_files": [],
        "failed_files": [],
        "total_text_chunks": 0,
        "total_visual_content": 0,
        "visual_indexing_stats": {
            "charts_indexed": 0,
            "tables_indexed": 0,
            "charts_failed": 0,
            "tables_failed": 0
        },
        "status": "Indexation en cours...",
        "total_files_found": len(pdf_paths)
    }

    if not pdf_paths:
        stats["status"] = "Aucun fichier PDF trouvé"
        return stats

    # Préparer le retriever
    async with retrieval.make_retriever(config) as retriever:
        vectorstore = retriever.vectorstore
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # 1. Indexer le contenu textuel des PDFs
        print("📄 Indexation du contenu textuel...")
        total_text_chunks = 0
        
        for idx, pdf in enumerate(pdf_paths, 1):
            try:
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
                
                # Indexer le batch textuel
                if texts:
                    await asyncio.to_thread(
                        vectorstore.add_texts,
                        texts=texts,
                        metadatas=metas,
                        ids=ids
                    )
                    total_text_chunks += len(texts)
                
                state.mark_text_processed(str(pdf))
                stats["processed_files"].append(pdf.name)
                
                print(f"   📄 PDF {idx}/{len(pdf_paths)}: {pdf.name} ({len(texts)} chunks)")

            except Exception as e:
                logging.error(f"Erreur traitement PDF {pdf.name}: {e}")
                stats["failed_files"].append(pdf.name)

        stats["total_text_chunks"] = total_text_chunks
        print(f"✅ Contenu textuel indexé: {total_text_chunks} chunks")

        # 2. Indexer le contenu visuel SI activé ET extraction complétée
        visual_extraction_completed = getattr(state, 'visual_extraction_completed', False)
        
        # DEBUG: Ajouter des logs pour diagnostiquer
        print(f"🔍 DEBUG - Vérification de l'indexation visuelle:")
        print(f"   cfg.enable_visual_indexing: {cfg.enable_visual_indexing}")
        print(f"   visual_extraction_completed: {visual_extraction_completed}")
        print(f"   state has visual_extraction_completed: {hasattr(state, 'visual_extraction_completed')}")
        
        
        if cfg.enable_visual_indexing and (Path(cfg.chart_index_path).exists() or Path(cfg.table_index_path).exists()):
            print("🎨 Indexation du contenu visuel...")
            
            visual_stats = await index_visual_content(
                vectorstore=vectorstore,
                cfg=cfg
            )
            
            stats["visual_indexing_stats"] = visual_stats
            stats["total_visual_content"] = (
                visual_stats["charts_indexed"] + visual_stats["tables_indexed"]
            )
            
            # Marquer l'indexation visuelle comme terminée
            for pdf_path in stats["processed_files"]:
                state.mark_image_processed(pdf_path)
            
            print(f"✅ Contenu visuel indexé:")
            print(f"   📊 {visual_stats['charts_indexed']} graphiques")
            print(f"   📋 {visual_stats['tables_indexed']} tableaux")
        
        else:
            # Message plus détaillé selon la cause
            if not cfg.enable_visual_indexing:
                print("⚠️ Indexation visuelle désactivée dans la configuration")
            elif not visual_extraction_completed:
                print("⚠️ Indexation visuelle ignorée (extraction non complétée)")
                print("   Vérifiez les logs d'extraction pour identifier les problèmes")

        # 3. Rapport final
        #total_content = stats["total_text_chunks"] + stats["total_visual_content"]
        
        # Dans la fonction index_local_pdfs, corriger le rapport final:
        stats["processed_files"] = [str(path) for path in pdf_paths if str(path) not in stats["failed_files"]]
        
        # 3. Rapport final
        total_content = stats["total_text_chunks"] + stats["total_visual_content"]
        
        print("="*60)
        print("📊 RAPPORT D'INDEXATION FINAL")
        print("="*60)
        print(f"📄 Chunks de texte indexés: {stats['total_text_chunks']}")
        print(f"📊 Graphiques indexés: {stats['visual_indexing_stats']['charts_indexed']}")
        print(f"📋 Tableaux indexés: {stats['visual_indexing_stats']['tables_indexed']}")
        print(f"🎯 Total contenu indexé: {total_content} éléments")
        print(f"✅ PDFs traités: {len(stats['processed_files'])}")
        
        if stats["failed_files"]:
            print(f"❌ PDFs échoués: {len(stats['failed_files'])}")
        
        # Statistiques d'échecs visuels
        visual_stats = stats["visual_indexing_stats"]
        if visual_stats["charts_failed"] > 0 or visual_stats["tables_failed"] > 0:
            print(f"⚠️ Échecs visuels: {visual_stats['charts_failed']} graphiques, {visual_stats['tables_failed']} tableaux")
        
        print("="*60)
        
        stats["status"] = f"Indexation terminée - {total_content} éléments indexés"
        return stats




async def index_visual_content(vectorstore, cfg: IndexConfiguration) -> Dict[str, int]:
    """
    Indexe le contenu visuel (graphiques et tableaux) extrait.
    
    Args:
        vectorstore: Vector store Pinecone
        cfg: Configuration d'indexation
        
    Returns:
        Statistiques d'indexation visuelle
    """
    import pandas as pd
    
    stats = {
        "charts_indexed": 0,
        "tables_indexed": 0,
        "charts_failed": 0,
        "tables_failed": 0
    }
    
    # 1. Indexer les graphiques
    charts_path = Path(cfg.chart_index_path)
    if charts_path.exists():
        try:
            charts_df = pd.read_csv(charts_path)
            print(f"   📊 Indexation de {len(charts_df)} graphiques...")
            
            # Traiter par petits batches
            batch_size = getattr(cfg, 'visual_batch_size', 5)
            total_processed = 0
            
            for i in range(0, len(charts_df), batch_size):
                batch = charts_df.iloc[i:i+batch_size]
                texts, metadatas, ids = [], [], []
                
                print(f"      🔄 Traitement batch {i//batch_size + 1}/{(len(charts_df)-1)//batch_size + 1}")
                
                for idx, row in batch.iterrows():
                    try:
                        image_path = Path(row['image_path'])
                        
                        print(f"         📸 Extraction texte: {image_path.name}")
                        
                        # Extraire le texte de l'image
                        extracted_text = await extract_text_from_image_content(
                            image_path, cfg.api_key
                        )
                        
                        if extracted_text and len(extracted_text.strip()) > 10:
                            # Créer le contenu du document
                            content_parts = [
                                f"Type: Graphique ANSD",
                                f"Caption: {row.get('caption', 'Non spécifiée')}",
                                f"Source PDF: {Path(row.get('pdf_path', '')).name}",
                                f"Page: {row.get('page', 'N/A')}",
                                f"Contenu extrait: {extracted_text}"
                            ]
                            
                            content = "\n".join(content_parts)
                            
                            # Métadonnées enrichies
                            metadata = {
                                "type": "visual_chart",
                                "source_type": "visual",
                                "image_id": str(row.get('image_id', f"chart_{idx}")),
                                "pdf_path": str(row.get('pdf_path', '')),
                                "pdf_name": Path(row.get('pdf_path', '')).name,
                                "page": int(row.get('page', 0)),
                                "image_path": str(image_path),
                                "caption": str(row.get('caption', '')),
                                "indexed_at": datetime.utcnow().isoformat(),
                                "content_source": "vision_api_extraction",
                                "content_length": len(extracted_text)
                            }
                            
                            # ID unique pour éviter les doublons
                            doc_id = f"visual_chart_{row.get('image_id', idx)}"
                            
                            texts.append(content)
                            metadatas.append(metadata)
                            ids.append(doc_id)
                            print(f"         ✅ Texte extrait: {len(extracted_text)} caractères")
                            
                        else:
                            print(f"         ⚠️ Texte vide ou trop court: {image_path.name}")
                            stats["charts_failed"] += 1
                            
                    except Exception as e:
                        logging.error(f"Erreur traitement graphique {idx}: {e}")
                        print(f"         ❌ Erreur: {e}")
                        stats["charts_failed"] += 1
                        continue
                
                # Indexer le batch
                if texts:
                    try:
                        await asyncio.to_thread(
                            vectorstore.add_texts,
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        stats["charts_indexed"] += len(texts)
                        total_processed += len(texts)
                        print(f"      ✅ Batch graphiques indexé: {len(texts)} éléments")
                    except Exception as e:
                        logging.error(f"Erreur indexation batch graphiques: {e}")
                        print(f"      ❌ Erreur indexation batch: {e}")
                        stats["charts_failed"] += len(texts)
                else:
                    print(f"      ⚠️ Aucun texte valide dans ce batch")
                
                # Petite pause pour éviter les limites de taux API
                await asyncio.sleep(1)
            
            print(f"   📊 Graphiques traités: {total_processed}/{len(charts_df)}")
                        
        except Exception as e:
            logging.error(f"Erreur indexation graphiques: {e}")
            print(f"   ❌ Erreur générale graphiques: {e}")
    
    # 2. Indexer les tableaux
    tables_path = Path(cfg.table_index_path)
    if tables_path.exists():
        try:
            tables_df = pd.read_csv(tables_path)
            print(f"   📋 Indexation de {len(tables_df)} tableaux...")
            
            # Traiter par petits batches
            batch_size = getattr(cfg, 'visual_batch_size', 10)  # Plus rapide pour les tableaux
            total_processed = 0
            
            for i in range(0, len(tables_df), batch_size):
                batch = tables_df.iloc[i:i+batch_size]
                texts, metadatas, ids = [], [], []
                
                print(f"      🔄 Traitement batch tableaux {i//batch_size + 1}/{(len(tables_df)-1)//batch_size + 1}")
                
                for idx, row in batch.iterrows():
                    try:
                        table_path = Path(row['table_path'])
                        
                        print(f"         📊 Extraction tableau: {table_path.name}")
                        
                        # Extraire le texte du tableau
                        extracted_text = await extract_text_from_table_content(table_path)
                        
                        if extracted_text and len(extracted_text.strip()) > 20:
                            # Créer le contenu du document
                            content_parts = [
                                f"Type: Tableau ANSD",
                                f"Caption: {row.get('caption', 'Non spécifiée')}",
                                f"Source PDF: {Path(row.get('pdf_path', '')).name}",
                                f"Page: {row.get('page', 'N/A')}",
                                f"Données du tableau: {extracted_text}"
                            ]
                            
                            content = "\n".join(content_parts)
                            
                            # Métadonnées enrichies
                            metadata = {
                                "type": "visual_table",
                                "source_type": "visual",
                                "table_id": str(row.get('table_id', f"table_{idx}")),
                                "pdf_path": str(row.get('pdf_path', '')),
                                "pdf_name": Path(row.get('pdf_path', '')).name,
                                "page": int(row.get('page', 0)),
                                "table_path": str(table_path),
                                "caption": str(row.get('caption', '')),
                                "indexed_at": datetime.utcnow().isoformat(),
                                "content_source": "csv_parsing",
                                "content_length": len(extracted_text)
                            }
                            
                            # ID unique
                            doc_id = f"visual_table_{row.get('table_id', idx)}"
                            
                            texts.append(content)
                            metadatas.append(metadata)
                            ids.append(doc_id)
                            print(f"         ✅ Tableau traité: {len(extracted_text)} caractères")
                            
                        else:
                            print(f"         ⚠️ Tableau vide ou trop petit: {table_path.name}")
                            stats["tables_failed"] += 1
                            
                    except Exception as e:
                        logging.error(f"Erreur traitement tableau {idx}: {e}")
                        print(f"         ❌ Erreur: {e}")
                        stats["tables_failed"] += 1
                        continue
                
                # Indexer le batch
                if texts:
                    try:
                        await asyncio.to_thread(
                            vectorstore.add_texts,
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        stats["tables_indexed"] += len(texts)
                        total_processed += len(texts)
                        print(f"      ✅ Batch tableaux indexé: {len(texts)} éléments")
                    except Exception as e:
                        logging.error(f"Erreur indexation batch tableaux: {e}")
                        print(f"      ❌ Erreur indexation batch: {e}")
                        stats["tables_failed"] += len(texts)
                else:
                    print(f"      ⚠️ Aucun contenu valide dans ce batch")
            
            print(f"   📋 Tableaux traités: {total_processed}/{len(tables_df)}")
                        
        except Exception as e:
            logging.error(f"Erreur indexation tableaux: {e}")
            print(f"   ❌ Erreur générale tableaux: {e}")
    
    return stats




# Construction du graphe avec la nouvelle étape d'extraction
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)

# Nœuds
builder.add_node("check_index_config", check_index_config)
builder.add_node("extract_visual_content", extract_visual_content)
builder.add_node("index_local_pdfs", index_local_pdfs)

# Flux d'exécution
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "extract_visual_content")
builder.add_edge("extract_visual_content", "index_local_pdfs")
builder.add_edge("index_local_pdfs", END)

# Compilation
graph = builder.compile()
graph.name = "CompleteIndexGraphWithVisualExtraction"