# =============================================================================
# TEST SIMPLE AVEC LES CORRECTIONS
# =============================================================================

import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_visual_system_with_debug():
    """Test direct du système avec les corrections appliquées."""
    
    print("🧪 TEST SYSTÈME VISUEL AVEC DEBUG")
    print("=" * 50)
    
    try:
        # Importer votre graph corrigé
        from simple_rag.graph import graph
        
        config = {
            "configurable": {
                "model": "openai/gpt-4o-mini",
                "retrieval_k": 15,
            }
        }
        
        # Test avec la même question que vous avez utilisée
        test_question = "Graphique évolution de la population"
        
        print(f"🔍 Question test: {test_question}")
        print("-" * 30)
        
        input_data = {
            "messages": [{"role": "user", "content": test_question}]
        }
        
        print("⏳ Exécution du graphe avec debug...")
        result = await graph.ainvoke(input_data, config=config)
        
        if result and "messages" in result:
            response = result["messages"][-1].content
            
            print(f"\n📝 RÉPONSE GÉNÉRÉE:")
            print("=" * 30)
            print(response)
            print("=" * 30)
            
            # Analyser la réponse
            visual_mentions = 0
            for keyword in ['graphique', 'image', 'élément visuel', 'affiché', 'Image trouvée']:
                if keyword.lower() in response.lower():
                    visual_mentions += 1
            
            print(f"\n📊 ANALYSE:")
            print(f"   Longueur réponse: {len(response)} caractères")
            print(f"   Mentions visuelles: {visual_mentions}")
            
            if "✅ IMAGE" in response or "✅ Graphique affiché" in response:
                print("🎉 SUCCESS! Image détectée et affichée!")
            elif "⚠️ AUCUNE IMAGE TROUVÉE" in response:
                print("⚠️ Image détectée mais fichier manquant")
            elif "❌ Aucune image correspondante" in response:
                print("❌ Problème de correspondance fichier")
            else:
                print("🔍 Vérifiez les logs ci-dessus pour plus de détails")
        
        # Analyser l'état du graph si disponible
        if result and "visual_elements" in result:
            visual_elements = result["visual_elements"]
            has_visual = result.get("has_visual_content", False)
            
            print(f"\n🎯 ÉTAT DU GRAPH:")
            print(f"   Visual elements: {len(visual_elements)}")
            print(f"   Has visual content: {has_visual}")
            
            if visual_elements:
                print(f"   📊 Premier élément:")
                first_elem = visual_elements[0]
                metadata = first_elem.get('metadata', {})
                print(f"      Type: {first_elem.get('type', 'N/A')}")
                print(f"      PDF: {metadata.get('pdf_name', 'N/A')}")
                print(f"      Page: {metadata.get('page_num', metadata.get('page', 'N/A'))}")
                
                # Vérifier métadonnées d'image
                image_keys = {k: v for k, v in metadata.items() 
                             if any(term in k.lower() for term in ['image', 'path', 'file'])}
                if image_keys:
                    print(f"      🖼️ Métadonnées image: {image_keys}")
                else:
                    print(f"      ❌ Pas de métadonnées image")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

async def quick_image_match_test():
    """Test rapide pour voir la correspondance images/PDF."""
    
    print(f"\n🔍 TEST CORRESPONDANCE IMAGES")
    print("=" * 30)
    
    from pathlib import Path
    import glob
    
    # Analyser quelques images de votre dossier
    image_files = list(Path('images').glob('*.png'))[:5]
    
    for image_file in image_files:
        print(f"\n📸 {image_file.name}")
        
        # Extraire info du nom
        filename = image_file.stem
        
        # Chercher page
        page_match = None
        if '_p' in filename:
            parts = filename.split('_p')
            if len(parts) > 1 and parts[1][0].isdigit():
                page_match = parts[1].split('_')[0]
        
        # Chercher PDF
        pdf_match = None
        if 'Chapitre' in filename:
            pdf_parts = filename.split('_')
            for i, part in enumerate(pdf_parts):
                if 'Chapitre' in part:
                    # Prendre jusqu'à juillet2024
                    pdf_parts_slice = pdf_parts[i:]
                    for j, p in enumerate(pdf_parts_slice):
                        if 'juillet2024' in p:
                            pdf_match = '_'.join(pdf_parts_slice[:j+1])
                            break
                    break
        
        print(f"   PDF probable: {pdf_match}")
        print(f"   Page probable: {page_match}")
        
        # Cette image pourrait correspondre à un document Pinecone
        # avec pdf_name contenant pdf_match et page_num = page_match

if __name__ == "__main__":
    print("🇸🇳 TEST SYSTÈME VISUEL ANSD")
    print("=" * 60)
    
    async def run_tests():
        # D'abord analyser les images
        await quick_image_match_test()
        
        # Puis tester le système
        await test_visual_system_with_debug()
        
        print("\n" + "=" * 60)
        print("💡 INTERPRÉTATION DES RÉSULTATS:")
        print("✅ 'IMAGE AFFICHÉE' = Succès complet")
        print("⚠️ 'IMAGE TROUVÉE' = Détection OK, affichage à vérifier") 
        print("❌ 'AUCUNE IMAGE' = Problème de métadonnées ou chemins")
        print("🔍 'DEBUG VISUAL: 0' = Aucun élément visuel détecté")
    
    asyncio.run(run_tests())