#!/usr/bin/env python3
"""
Script de test pour l'indexation complète avec extraction et indexation visuelle.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Charger .env
load_dotenv()

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Vérifie les dépendances requises."""
    
    print("🔍 Vérification des dépendances...")
    
    missing = []
    
    # Vérifier PyMuPDF
    try:
        import fitz
        print("   ✅ PyMuPDF disponible")
    except ImportError:
        missing.append("PyMuPDF (pip install PyMuPDF)")
    
    # Vérifier Camelot
    try:
        import camelot
        print("   ✅ Camelot disponible")
    except ImportError:
        missing.append("camelot-py[cv] (pip install camelot-py[cv])")
    
    # Vérifier Pillow
    try:
        from PIL import Image
        print("   ✅ Pillow disponible")
    except ImportError:
        missing.append("Pillow (pip install Pillow)")
    
    # Vérifier pandas
    try:
        import pandas as pd
        print("   ✅ Pandas disponible")
    except ImportError:
        missing.append("pandas (pip install pandas)")
    
    if missing:
        print("\n❌ Dépendances manquantes:")
        for dep in missing:
            print(f"   • {dep}")
        return False
    
    print("✅ Toutes les dépendances sont satisfaites")
    return True


def check_environment():
    """Vérifie les variables d'environnement."""
    
    print("\n🔧 Vérification de l'environnement...")
    
    required_vars = {
        "OPENAI_API_KEY": "Clé API OpenAI (pour extraction visuelle)",
        "PINECONE_API_KEY": "Clé API Pinecone",
        "INDEX_API_KEY": "Clé pour l'indexation (sécurité)"
    }
    
    missing = []
    
    for var, description in required_vars.items():
        if os.getenv(var):
            masked = os.getenv(var)[:8] + "..." if len(os.getenv(var)) > 8 else "***"
            print(f"   ✅ {var}: {masked}")
        else:
            missing.append(f"{var} ({description})")
    
    if missing:
        print("\n❌ Variables d'environnement manquantes:")
        for var in missing:
            print(f"   • {var}")
        return False
    
    print("✅ Environnement configuré correctement")
    return True


async def test_complete_indexing(pdf_directory: str):
    """
    Test complet du processus d'indexation.
    
    Args:
        pdf_directory: Dossier contenant les PDFs à traiter
    """
    print(f"\n🚀 TEST D'INDEXATION COMPLÈTE")
    print(f"📁 Dossier source: {pdf_directory}")
    print("="*60)
    
    try:
        # Import des modules nécessaires
        from index_graph.graph import graph as index_graph
        from index_graph.state import IndexState
        
        # Vérifier que le dossier existe
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            print(f"❌ Dossier non trouvé: {pdf_directory}")
            return False
        
        # Compter les PDFs
        pdf_files = list(pdf_path.glob("**/*.pdf"))
        print(f"📄 {len(pdf_files)} fichiers PDF trouvés")
        
        if len(pdf_files) == 0:
            print("⚠️ Aucun fichier PDF trouvé dans le dossier")
            return False
        
        # Afficher les premiers PDFs
        print("📋 Exemples de fichiers:")
        for pdf in pdf_files[:3]:
            print(f"   • {pdf.name}")
        if len(pdf_files) > 3:
            print(f"   ... et {len(pdf_files) - 3} autres")
        
        # Préparer l'état initial
        initial_state = IndexState(
            pdf_root=str(pdf_directory),
            status="Initialisation",
            processed_text_files=[],
            failed_text_files=[],
            processed_image_files=[],
            failed_image_files=[],
            total_text_chunks=0,
            total_text_files_found=len(pdf_files),
            total_images_indexed=0
        )
        
        # Configuration
        config = {
            "configurable": {
                "retriever_provider": "pinecone",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
                "enable_visual_indexing": True,
                "chart_index_path": "charts_index.csv",
                "table_index_path": "tables_index.csv",
                "images_dir": "images",
                "tables_dir": "tables",
                "visual_batch_size": 3,  # Petite taille pour les tests
                "max_vision_retries": 2
            }
        }
        
        print("\n🎯 Lancement de l'indexation complète...")
        print("   1️⃣ Vérification de la configuration")
        print("   2️⃣ Extraction du contenu visuel")
        print("   3️⃣ Indexation textuelle et visuelle")
        
        # Exécuter le graphe complet
        result = await index_graph.ainvoke(initial_state, config=config)
        
        # Analyser les résultats
        print("\n📊 RÉSULTATS DE L'INDEXATION:")
        print("="*60)
        
        if result:
            print(f"✅ Statut: {result.get('status', 'Terminé')}")
            print(f"📄 Fichiers traités: {len(result.get('processed_files', []))}")
            print(f"📝 Chunks de texte: {result.get('total_text_chunks', 0)}")
            
            visual_stats = result.get('visual_indexing_stats', {})
            print(f"📊 Graphiques indexés: {visual_stats.get('charts_indexed', 0)}")
            print(f"📋 Tableaux indexés: {visual_stats.get('tables_indexed', 0)}")
            
            total_visual = visual_stats.get('charts_indexed', 0) + visual_stats.get('tables_indexed', 0)
            total_content = result.get('total_text_chunks', 0) + total_visual
            print(f"🎯 Total contenu indexé: {total_content} éléments")
            
            # Échecs
            failed_files = result.get('failed_files', [])
            if failed_files:
                print(f"❌ Fichiers échoués: {len(failed_files)}")
                for failed in failed_files[:3]:
                    print(f"   • {failed}")
            
            charts_failed = visual_stats.get('charts_failed', 0)
            tables_failed = visual_stats.get('tables_failed', 0)
            if charts_failed or tables_failed:
                print(f"⚠️ Échecs visuels: {charts_failed} graphiques, {tables_failed} tableaux")
        
        # Vérifier les fichiers créés
        print(f"\n📁 FICHIERS CRÉÉS:")
        
        charts_index = Path("charts_index.csv")
        if charts_index.exists():
            import pandas as pd
            charts_df = pd.read_csv(charts_index)
            print(f"   📊 charts_index.csv: {len(charts_df)} entrées")
        else:
            print(f"   ❌ charts_index.csv: non créé")
        
        tables_index = Path("tables_index.csv")
        if tables_index.exists():
            import pandas as pd
            tables_df = pd.read_csv(tables_index)
            print(f"   📋 tables_index.csv: {len(tables_df)} entrées")
        else:
            print(f"   ❌ tables_index.csv: non créé")
        
        images_dir = Path("images")
        if images_dir.exists():
            images = list(images_dir.glob("**/*.png"))
            print(f"   🖼️ images/: {len(images)} fichiers")
        else:
            print(f"   ❌ images/: dossier non créé")
        
        tables_dir = Path("tables")
        if tables_dir.exists():
            tables = list(tables_dir.glob("**/*.csv"))
            print(f"   📊 tables/: {len(tables)} fichiers")
        else:
            print(f"   ❌ tables/: dossier non créé")
        
        print("="*60)
        print("✅ Test d'indexation complète terminé")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DU TEST:")
        print(f"   {type(e).__name__}: {e}")
        
        import traceback
        print(f"\n🔍 Détails de l'erreur:")
        traceback.print_exc()
        
        return False


async def test_search_functionality():
    """Test de la fonctionnalité de recherche avec contenu visuel."""
    
    print(f"\n🔍 TEST DE RECHERCHE HYBRIDE")
    print("="*40)
    
    try:
        from shared.retrieval import make_retriever
        
        # Configuration pour la recherche
        config = {
            "configurable": {
                "retriever_provider": "pinecone",
                "embedding_model": "openai/text-embedding-3-small",
                "search_kwargs": {"k": 10}
            }
        }
        
        # Test queries
        test_queries = [
            "population du Sénégal",
            "graphique démographique",
            "tableau statistiques",
            "données RGPH"
        ]
        
        async with make_retriever(config) as retriever:
            for query in test_queries:
                print(f"\n🔍 Recherche: '{query}'")
                
                results = await retriever.ainvoke(query)
                print(f"   📊 {len(results)} résultats trouvés")
                
                # Analyser les types de résultats
                text_results = [r for r in results if r.metadata.get("source_type") != "visual"]
                visual_results = [r for r in results if r.metadata.get("source_type") == "visual"]
                
                print(f"   📄 Texte: {len(text_results)} | 🎨 Visuel: {len(visual_results)}")
                
                # Montrer quelques exemples
                for i, result in enumerate(results[:2], 1):
                    source_type = result.metadata.get("source_type", "text")
                    content_type = result.metadata.get("type", "unknown")
                    content_preview = result.page_content[:100] + "..."
                    
                    print(f"   {i}. [{source_type}] {content_type}: {content_preview}")
        
        print("✅ Test de recherche terminé")
        
    except Exception as e:
        print(f"❌ Erreur test recherche: {e}")


def main():
    """Fonction principale."""
    
    print("🇸🇳 TEST COMPLET D'INDEXATION VISUELLE ANSD")
    print("="*60)
    
    # Vérifications préliminaires
    if not check_dependencies():
        print("\n💡 Installez les dépendances manquantes et relancez le test")
        return
    
    if not check_environment():
        print("\n💡 Configurez les variables d'environnement dans votre fichier .env")
        return
    
    # Demander le dossier PDF
    pdf_directory = input("\n📁 Chemin vers le dossier contenant les PDFs: ").strip()
    
    if not pdf_directory:
        print("❌ Aucun dossier spécifié")
        return
    
    # Menu d'options
    print(f"\nOptions de test:")
    print("1. Test complet (extraction + indexation)")
    print("2. Test de recherche uniquement")
    print("3. Les deux")
    
    choice = input("Choix (1-3): ").strip()
    
    async def run_tests():
        if choice in ["1", "3"]:
            success = await test_complete_indexing(pdf_directory)
            if not success:
                return
        
        if choice in ["2", "3"]:
            await test_search_functionality()
    
    # Exécuter les tests
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")


if __name__ == "__main__":
    main()