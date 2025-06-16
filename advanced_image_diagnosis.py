# =============================================================================
# DIAGNOSTIC AVANCÉ POUR LES MÉTADONNÉES D'IMAGES DANS PINECONE
# =============================================================================

import asyncio
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

async def diagnose_pinecone_images():
    """Diagnostic approfondi des métadonnées d'images dans Pinecone."""
    
    print("🔍 DIAGNOSTIC AVANCÉ DES IMAGES DANS PINECONE")
    print("=" * 60)
    
    try:
        from src.shared import retrieval
        
        config = {
            "configurable": {
                "search_kwargs": {"k": 30}  # Plus de documents pour analysis
            }
        }
        
        # Requêtes spécifiques pour trouver les éléments visuels
        visual_queries = [
            "graphique évolution population",
            "visual_chart",
            "Chapitre 1 ETAT STRUCTURE POPULATION",
            "page 5"  # Votre graphique était page 5
        ]
        
        all_visual_docs = []
        
        for query in visual_queries:
            print(f"\n🔍 Recherche: '{query}'")
            
            async with retrieval.make_retriever(config) as retriever:
                documents = await retriever.ainvoke(query, config)
                
                print(f"📄 Documents trouvés: {len(documents)}")
                
                for i, doc in enumerate(documents[:5], 1):
                    metadata = doc.metadata
                    content_preview = doc.page_content[:150].replace('\n', ' ')
                    
                    print(f"\n  📄 Document {i}:")
                    print(f"    PDF: {metadata.get('pdf_name', 'N/A')}")
                    print(f"    Page: {metadata.get('page_num', metadata.get('page', 'N/A'))}")
                    print(f"    Type: {metadata.get('type', 'N/A')}")
                    print(f"    Source: {metadata.get('source', 'N/A')}")
                    print(f"    Contenu: {content_preview}...")
                    
                    # Vérifier toutes les clés qui pourraient contenir un chemin d'image
                    image_keys = []
                    for key, value in metadata.items():
                        if any(term in key.lower() for term in ['image', 'path', 'file', 'png', 'jpg']):
                            image_keys.append(f"{key}: {value}")
                    
                    if image_keys:
                        print(f"    🖼️ Clés d'image: {', '.join(image_keys)}")
                    
                    # Vérifier si c'est un élément visuel potentiel
                    is_visual = False
                    visual_indicators = []
                    
                    if metadata.get('type') in ['visual_chart', 'visual_table', 'image']:
                        is_visual = True
                        visual_indicators.append(f"type={metadata.get('type')}")
                    
                    if 'graphique' in content_preview.lower() or 'figure' in content_preview.lower():
                        is_visual = True
                        visual_indicators.append("contenu_graphique")
                    
                    if any(term in str(metadata.get('source', '')).lower() for term in ['.png', '.jpg', 'image']):
                        is_visual = True
                        visual_indicators.append("source_image")
                    
                    if is_visual:
                        print(f"    🎨 ÉLÉMENT VISUEL DÉTECTÉ: {', '.join(visual_indicators)}")
                        all_visual_docs.append({
                            'doc': doc,
                            'metadata': metadata,
                            'indicators': visual_indicators
                        })
                    else:
                        print(f"    📝 Document textuel")
        
        # Analyser les éléments visuels trouvés
        print(f"\n" + "=" * 60)
        print(f"📊 ANALYSE DES ÉLÉMENTS VISUELS TROUVÉS")
        print(f"=" * 60)
        print(f"Total d'éléments visuels détectés: {len(all_visual_docs)}")
        
        if all_visual_docs:
            print(f"\n🔍 ANALYSE DÉTAILLÉE:")
            
            for i, visual_doc in enumerate(all_visual_docs[:3], 1):
                metadata = visual_doc['metadata']
                print(f"\n📊 Élément visuel {i}:")
                print(f"   PDF: {metadata.get('pdf_name', 'N/A')}")
                print(f"   Page: {metadata.get('page_num', metadata.get('page', 'N/A'))}")
                print(f"   Indicateurs: {', '.join(visual_doc['indicators'])}")
                
                # Tenter de trouver l'image correspondante
                pdf_name = metadata.get('pdf_name', '')
                page_num = metadata.get('page_num', metadata.get('page', ''))
                
                if pdf_name and page_num:
                    # Chercher l'image correspondante dans le dossier images/
                    potential_images = find_matching_image(pdf_name, page_num)
                    
                    if potential_images:
                        print(f"   ✅ Images correspondantes trouvées:")
                        for img in potential_images[:2]:
                            print(f"      • {img}")
                    else:
                        print(f"   ❌ Aucune image correspondante trouvée")
                        print(f"   💡 Recherche pattern: *{Path(pdf_name).stem}*{page_num}*.png")
        
        else:
            print(f"❌ Aucun élément visuel trouvé dans Pinecone")
            print(f"💡 Cela signifie que vos images ne sont pas indexées avec les bonnes métadonnées")
    
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")
        import traceback
        traceback.print_exc()

def find_matching_image(pdf_name, page_num):
    """Trouve les images qui correspondent au PDF et à la page."""
    
    import glob
    
    # Nettoyer le nom du PDF
    pdf_stem = Path(pdf_name).stem if '.' in pdf_name else pdf_name
    pdf_clean = pdf_stem.replace(' ', '*').replace('_', '*')
    
    # Patterns de recherche
    patterns = [
        f"images/*{pdf_clean}*{page_num}*.png",
        f"images/*{pdf_clean}*p{page_num}*.png",
        f"images/{pdf_clean}*page*{page_num}*.png",
        f"images/*{pdf_stem}*{page_num}*.png",
        f"images/*page*{page_num}*.png"
    ]
    
    found_images = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        found_images.extend(matches)
    
    # Supprimer les doublons
    return list(set(found_images))

async def test_specific_image():
    """Test avec une image spécifique de votre dossier."""
    
    print(f"\n🧪 TEST AVEC UNE IMAGE SPÉCIFIQUE")
    print("=" * 40)
    
    # Prendre la première image de votre dossier
    image_files = list(Path('images').glob('*.png'))
    if image_files:
        test_image = image_files[0]
        print(f"📸 Test avec: {test_image.name}")
        
        # Extraire les informations du nom de fichier
        filename = test_image.stem
        parts = filename.split('_')
        
        print(f"🔍 Analyse du nom de fichier:")
        print(f"   Nom complet: {filename}")
        print(f"   Parties: {parts}")
        
        # Essayer de deviner le PDF et la page
        potential_pdf = None
        potential_page = None
        
        for part in parts:
            if part.startswith('p') and part[1:].isdigit():
                potential_page = part[1:]
            elif 'page' in part.lower():
                potential_page = part.lower().replace('page', '')
            elif part.startswith('Chapitre'):
                potential_pdf = ' '.join(parts[:3])  # Prendre les 3 premières parties
        
        if potential_pdf:
            print(f"   PDF probable: {potential_pdf}")
        if potential_page:
            print(f"   Page probable: {potential_page}")
        
        # Chercher dans Pinecone
        if potential_pdf:
            print(f"\n🔍 Recherche dans Pinecone...")
            
            try:
                from shared import retrieval
                
                config = {"configurable": {"search_kwargs": {"k": 10}}}
                
                async with retrieval.make_retriever(config) as retriever:
                    documents = await retriever.ainvoke(potential_pdf, config)
                    
                    print(f"📄 Documents trouvés pour '{potential_pdf}': {len(documents)}")
                    
                    for doc in documents[:3]:
                        metadata = doc.metadata
                        doc_page = metadata.get('page_num', metadata.get('page', 'N/A'))
                        doc_pdf = metadata.get('pdf_name', 'N/A')
                        
                        print(f"   📄 {doc_pdf}, page {doc_page}")
                        
                        if str(doc_page) == str(potential_page):
                            print(f"   ✅ CORRESPONDANCE TROUVÉE !")
                            print(f"   💡 Ce document devrait pointer vers: {test_image}")
                            
                            # Vérifier les métadonnées d'image
                            image_meta = []
                            for key, value in metadata.items():
                                if 'image' in key.lower() or 'path' in key.lower():
                                    image_meta.append(f"{key}={value}")
                            
                            if image_meta:
                                print(f"   🖼️ Métadonnées d'image: {', '.join(image_meta)}")
                            else:
                                print(f"   ❌ Aucune métadonnée d'image trouvée")
            
            except Exception as e:
                print(f"   ❌ Erreur recherche: {e}")

if __name__ == "__main__":
    async def run_full_diagnosis():
        await diagnose_pinecone_images()
        await test_specific_image()
        
        print(f"\n" + "=" * 60)
        print(f"💡 PROCHAINES ÉTAPES:")
        print(f"1. Si des éléments visuels sont détectés mais sans image_path:")
        print(f"   → Le problème est dans l'indexation")
        print(f"2. Si aucun élément visuel n'est détecté:")
        print(f"   → Les images ne sont pas indexées du tout")
        print(f"3. Si tout semble correct:")
        print(f"   → Le problème est dans l'affichage")
    
    asyncio.run(run_full_diagnosis())