# =============================================================================
# SCRIPT DE DIAGNOSTIC POUR LES ÉLÉMENTS VISUELS DANS PINECONE
# =============================================================================

import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

async def diagnose_visual_elements():
    """Diagnostic pour identifier pourquoi les éléments visuels ne sont pas détectés."""
    
    print("🔍 DIAGNOSTIC DES ÉLÉMENTS VISUELS DANS PINECONE")
    print("=" * 60)
    
    try:
        # Import du retrieval système
        from shared import retrieval
        
        # Configuration de test
        config = {
            "configurable": {
                "search_kwargs": {"k": 20}  # Plus de documents pour diagnostic
            }
        }
        
        # Questions de test pour les visuels
        test_questions = [
            "population Sénégal 2023",
            "graphique RGPH",
            "tableau démographique",
            "données ANSD",
            "statistiques population"
        ]
        
        for question in test_questions:
            print(f"\n🔍 Test: {question}")
            print("-" * 30)
            
            # Récupérer les documents
            async with retrieval.make_retriever(config) as retriever:
                documents = await retriever.ainvoke(question, config)
                
                print(f"📄 Documents récupérés: {len(documents)}")
                
                # Analyser chaque document
                visual_count = 0
                text_count = 0
                
                for i, doc in enumerate(documents, 1):
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    content_preview = doc.page_content[:100].replace('\n', ' ') if hasattr(doc, 'page_content') else str(doc)[:100]
                    
                    print(f"\n  📄 Document {i}:")
                    print(f"    Contenu: {content_preview}...")
                    
                    # Afficher les métadonnées importantes
                    important_keys = ['pdf_name', 'page_num', 'source', 'type', 'image_path', 'chart_type', 'is_table', 'visual_type', 'content_type']
                    
                    print(f"    Métadonnées:")
                    for key in important_keys:
                        if key in metadata:
                            print(f"      {key}: {metadata[key]}")
                    
                    # Détecter le type
                    is_visual = detect_visual_element(doc, metadata)
                    
                    if is_visual:
                        visual_count += 1
                        print(f"    🎨 TYPE: ÉLÉMENT VISUEL")
                    else:
                        text_count += 1
                        print(f"    📝 TYPE: TEXTE")
                
                print(f"\n📊 Résumé: {visual_count} visuels, {text_count} texte")
                
                if visual_count == 0:
                    print("⚠️ AUCUN ÉLÉMENT VISUEL DÉTECTÉ")
                    print("💡 Vérifiez les métadonnées dans Pinecone")
        
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")
        import traceback
        traceback.print_exc()

def detect_visual_element(doc, metadata):
    """Fonction améliorée pour détecter les éléments visuels."""
    
    # Méthode 1: Vérifier les métadonnées explicites
    visual_indicators = [
        'image_path', 'chart_type', 'visual_type', 'is_table', 
        'table_data', 'content_type', 'chart_category'
    ]
    
    for indicator in visual_indicators:
        if indicator in metadata:
            print(f"      ✅ Détecté via métadonnée: {indicator}")
            return True
    
    # Méthode 2: Vérifier le type de document
    doc_type = metadata.get('type', '')
    if doc_type in ['visual_chart', 'visual_table', 'image', 'chart', 'table']:
        print(f"      ✅ Détecté via type: {doc_type}")
        return True
    
    # Méthode 3: Vérifier le nom du fichier source
    source = metadata.get('source', '')
    pdf_name = metadata.get('pdf_name', '')
    
    visual_file_patterns = ['.png', '.jpg', '.jpeg', '.csv', 'chart', 'table', 'graph']
    
    for pattern in visual_file_patterns:
        if pattern in source.lower() or pattern in pdf_name.lower():
            print(f"      ✅ Détecté via fichier: {pattern}")
            return True
    
    # Méthode 4: Analyser le contenu textuel
    if hasattr(doc, 'page_content'):
        content = doc.page_content.lower()
        
        # Indicateurs de tableau
        table_patterns = ['|', '\t', 'total', 'sous-total', 'colonnes:', 'ligne ']
        table_indicators = sum(1 for pattern in table_patterns if pattern in content)
        
        if table_indicators >= 2:
            print(f"      ✅ Détecté comme tableau (indicateurs: {table_indicators})")
            return True
        
        # Indicateurs de graphique
        chart_keywords = ['graphique', 'figure', 'diagramme', 'courbe', 'histogramme', 'secteur']
        if any(keyword in content for keyword in chart_keywords):
            print(f"      ✅ Détecté comme graphique via contenu")
            return True
    
    return False

async def test_specific_visual_query():
    """Test avec une requête spécifique pour les visuels."""
    
    print("\n🎯 TEST SPÉCIFIQUE POUR LES VISUELS")
    print("=" * 40)
    
    try:
        from shared import retrieval
        
        # Configuration élargie
        config = {
            "configurable": {
                "search_kwargs": {"k": 30}  # Beaucoup plus de documents
            }
        }
        
        # Requêtes ciblées visuels
        visual_queries = [
            "image graphique chart",
            "tableau csv données",
            "figure diagramme", 
            "png jpg image",
            "visualisation statistique"
        ]
        
        for query in visual_queries:
            print(f"\n🔍 Requête visuelle: {query}")
            
            async with retrieval.make_retriever(config) as retriever:
                documents = await retriever.ainvoke(query, config)
                
                visual_docs = []
                
                for doc in documents:
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    # Analyse détaillée
                    if detect_visual_element(doc, metadata):
                        visual_docs.append(doc)
                
                print(f"📊 Éléments visuels trouvés: {len(visual_docs)}")
                
                # Afficher les premiers
                for i, doc in enumerate(visual_docs[:3], 1):
                    metadata = doc.metadata
                    print(f"  {i}. {metadata.get('pdf_name', 'Unknown')} - {metadata.get('type', 'No type')}")
        
    except Exception as e:
        print(f"❌ Erreur test visuel: {e}")

async def check_pinecone_metadata_structure():
    """Vérifie la structure des métadonnées dans Pinecone."""
    
    print("\n🔬 ANALYSE DE LA STRUCTURE DES MÉTADONNÉES")
    print("=" * 50)
    
    try:
        from shared import retrieval
        
        config = {
            "configurable": {
                "search_kwargs": {"k": 50}  # Large échantillon
            }
        }
        
        async with retrieval.make_retriever(config) as retriever:
            documents = await retriever.ainvoke("ANSD données", config)
            
            # Analyser toutes les clés de métadonnées présentes
            all_metadata_keys = set()
            type_counts = {}
            source_patterns = set()
            
            for doc in documents:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Collecter toutes les clés
                all_metadata_keys.update(metadata.keys())
                
                # Compter les types
                doc_type = metadata.get('type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                # Analyser les sources
                source = metadata.get('source', '')
                if source:
                    # Extraire l'extension
                    if '.' in source:
                        ext = source.split('.')[-1].lower()
                        source_patterns.add(ext)
            
            print(f"📊 Analyse de {len(documents)} documents")
            print(f"\n🔑 Clés de métadonnées trouvées:")
            for key in sorted(all_metadata_keys):
                print(f"  • {key}")
            
            print(f"\n📋 Types de documents:")
            for doc_type, count in sorted(type_counts.items()):
                print(f"  • {doc_type}: {count}")
            
            print(f"\n📁 Extensions de fichiers:")
            for ext in sorted(source_patterns):
                print(f"  • .{ext}")
            
            # Recommandations
            print(f"\n💡 RECOMMANDATIONS:")
            
            if 'image_path' in all_metadata_keys:
                print("  ✅ Métadonnée 'image_path' trouvée - images indexées")
            else:
                print("  ❌ Métadonnée 'image_path' manquante")
            
            if 'visual_chart' in type_counts or 'visual_table' in type_counts:
                print("  ✅ Types visuels trouvés dans l'index")
            else:
                print("  ❌ Aucun type visuel explicite trouvé")
            
            if 'png' in source_patterns or 'jpg' in source_patterns:
                print("  ✅ Fichiers images détectés")
            else:
                print("  ❌ Aucun fichier image détecté")
    
    except Exception as e:
        print(f"❌ Erreur analyse métadonnées: {e}")

if __name__ == "__main__":
    print("🇸🇳 DIAGNOSTIC ÉLÉMENTS VISUELS ANSD")
    print("=" * 80)
    
    async def run_all_diagnostics():
        await diagnose_visual_elements()
        await test_specific_visual_query()
        await check_pinecone_metadata_structure()
        
        print("\n" + "=" * 80)
        print("✅ DIAGNOSTIC TERMINÉ")
        print("\n💡 ÉTAPES SUIVANTES:")
        print("  1. Vérifiez les métadonnées dans les résultats ci-dessus")
        print("  2. Adaptez la fonction detect_visual_element() selon vos métadonnées")
        print("  3. Relancez le système avec la détection corrigée")
    
    asyncio.run(run_all_diagnostics())