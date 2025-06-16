# =============================================================================
# TEST SIMPLE POUR VÉRIFIER LA DÉTECTION DES VISUELS
# =============================================================================

import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_visual_detection_system():
    """Test simple pour vérifier que les visuels sont bien détectés."""
    
    print("🧪 TEST DE DÉTECTION DES ÉLÉMENTS VISUELS")
    print("=" * 60)
    
    try:
        # 1. D'abord, lancer le diagnostic
        print("🔍 ÉTAPE 1: Diagnostic des métadonnées...")
        await run_diagnostic()
        
        # 2. Ensuite, tester avec le système amélioré
        print("\n🔧 ÉTAPE 2: Test avec détection améliorée...")
        await test_enhanced_detection()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

async def run_diagnostic():
    """Lance le diagnostic pour voir les métadonnées."""
    
    try:
        from shared import retrieval
        
        config = {
            "configurable": {
                "search_kwargs": {"k": 10}
            }
        }
        
        async with retrieval.make_retriever(config) as retriever:
            documents = await retriever.ainvoke("population Sénégal 2023", config)
            
            print(f"📄 Documents récupérés: {len(documents)}")
            
            # Afficher les métadonnées des 3 premiers
            for i, doc in enumerate(documents[:3], 1):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                print(f"\n📄 Document {i}:")
                print(f"  Source: {metadata.get('source', 'Unknown')}")
                print(f"  PDF: {metadata.get('pdf_name', 'Unknown')}")
                print(f"  Type: {metadata.get('type', 'Unknown')}")
                
                # Chercher des indicateurs visuels
                visual_keys = []
                for key in metadata.keys():
                    if any(term in key.lower() for term in ['image', 'chart', 'table', 'visual', 'graph']):
                        visual_keys.append(f"{key}: {metadata[key]}")
                
                if visual_keys:
                    print(f"  🎨 Métadonnées visuelles: {', '.join(visual_keys)}")
                else:
                    print(f"  📝 Pas de métadonnées visuelles détectées")
        
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")

async def test_enhanced_detection():
    """Test avec les fonctions de détection améliorées."""
    
    try:
        # Importer les nouvelles fonctions
        from corrected_visual_detection import (
            extract_visual_elements, 
            analyze_visual_relevance_enhanced
        )
        
        from shared import retrieval
        
        config = {
            "configurable": {
                "search_kwargs": {"k": 15}
            }
        }
        
        # Questions de test
        test_questions = [
            "population du Sénégal 2023",
            "graphique démographique ANSD",
            "tableau données RGPH",
            "statistiques emploi ENES"
        ]
        
        for question in test_questions:
            print(f"\n🔍 Test: {question}")
            print("-" * 30)
            
            async with retrieval.make_retriever(config) as retriever:
                documents = await retriever.ainvoke(question, config)
                
                # Utiliser les nouvelles fonctions
                text_docs, visual_elements = extract_visual_elements(documents)
                relevant_visuals = analyze_visual_relevance_enhanced(visual_elements, question)
                
                print(f"📊 Résultat: {len(text_docs)} textuels, {len(visual_elements)} visuels bruts, {len(relevant_visuals)} visuels pertinents")
                
                # Afficher les visuels pertinents
                for i, visual in enumerate(relevant_visuals[:2], 1):
                    metadata = visual['metadata']
                    score = visual.get('relevance_score', 0)
                    print(f"  🎨 Visuel {i}: {metadata.get('pdf_name', 'Unknown')} (score: {score})")
                    print(f"     Type: {visual['type']}")
        
    except ImportError:
        print("❌ Impossible d'importer les nouvelles fonctions")
        print("💡 Assurez-vous d'avoir créé le fichier corrected_visual_detection.py")
    except Exception as e:
        print(f"❌ Erreur test amélioré: {e}")

async def test_with_current_graph():
    """Test avec votre graph.py actuel pour comparaison."""
    
    print("\n🔄 TEST AVEC LE GRAPH ACTUEL")
    print("-" * 30)
    
    try:
        # Importer votre graph actuel
        from simple_rag.graph import graph
        
        config = {
            "configurable": {
                "model": "openai/gpt-4o-mini",
                "retrieval_k": 10,
            }
        }
        
        # Test simple
        input_data = {
            "messages": [{"role": "user", "content": "Montrez-moi des graphiques sur la population du Sénégal"}]
        }
        
        print("⏳ Exécution du graphe...")
        result = await graph.ainvoke(input_data, config=config)
        
        if result and "messages" in result:
            response = result["messages"][-1].content
            
            # Vérifier si des visuels ont été mentionnés
            visual_indicators = ['graphique', 'tableau', 'image', 'figure', 'élément visuel']
            visual_mentions = sum(1 for indicator in visual_indicators if indicator in response.lower())
            
            print(f"📝 Réponse générée ({len(response)} caractères)")
            print(f"🎨 Mentions visuelles: {visual_mentions}")
            
            if visual_mentions > 0:
                print("✅ Le système semble détecter du contenu visuel")
            else:
                print("❌ Aucune mention de contenu visuel dans la réponse")
                print("💡 Utilisez les fonctions corrigées pour améliorer la détection")
        
    except Exception as e:
        print(f"❌ Erreur test graph: {e}")

if __name__ == "__main__":
    print("🇸🇳 TEST SYSTÈME VISUEL ANSD")
    print("=" * 80)
    
    async def run_all_tests():
        await test_visual_detection_system()
        await test_with_current_graph()
        
        print("\n" + "=" * 80)
        print("📋 RÉSUMÉ ET ACTIONS:")
        print("1. ✅ Lancez d'abord le diagnostic pour voir vos métadonnées")
        print("2. 🔧 Adaptez les fonctions de détection selon vos métadonnées")
        print("3. 🔄 Remplacez les fonctions dans votre graph.py")
        print("4. 🧪 Testez avec des questions incluant 'graphique' ou 'tableau'")
        print("5. 🎯 Vérifiez que les visuels sont bien affichés")
    
    asyncio.run(run_all_tests())