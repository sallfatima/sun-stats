# =============================================================================
# SCRIPT DE TEST POUR L'INTÉGRATION DES SUGGESTIONS
# =============================================================================

import asyncio
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

async def test_suggestions_integration():
    """Test simple pour vérifier que l'intégration fonctionne."""
    
    print("🧪 TEST D'INTÉGRATION DES SUGGESTIONS ANSD")
    print("=" * 60)
    
    try:
        # Importer le graphe modifié
        from simple_rag.graph import graph
        print("✅ Import du graphe réussi")
        
        # Vérifier que les nouvelles fonctions sont disponibles
        from simple_rag.graph import (
            generate_question_suggestions, 
            extract_topics_from_documents,
            extract_topics_from_response,
            generate_fallback_suggestions
        )
        print("✅ Import des nouvelles fonctions réussi")
        
        # Configuration de test
        config = {
            "configurable": {
                "model": "openai/gpt-4o-mini",  # ou votre modèle préféré
                "retrieval_k": 5,
            }
        }
        
        # Question de test
        test_question = "Quelle est la population du Sénégal selon le RGPH ?"
        
        print(f"\n🔍 Test avec la question: {test_question}")
        
        # Préparer l'input
        input_data = {
            "messages": [{"role": "user", "content": test_question}]
        }
        
        print("\n⏳ Exécution du graphe...")
        
        # Exécuter le graphe
        result = await graph.ainvoke(input_data, config=config)
        
        if result and "messages" in result and result["messages"]:
            response = result["messages"][-1].content
            
            print(f"\n✅ Réponse générée ({len(response)} caractères)")
            
            # Vérifier la présence de suggestions
            if "❓ QUESTIONS SUGGÉRÉES" in response or "questions suggérées" in response.lower():
                print("🎉 SUCCÈS : Suggestions de questions détectées !")
                
                # Afficher un aperçu
                lines = response.split('\n')
                suggestion_started = False
                suggestion_count = 0
                
                print("\n🔮 Aperçu des suggestions :")
                for line in lines:
                    if "questions suggérées" in line.lower() or "❓" in line:
                        suggestion_started = True
                        continue
                    
                    if suggestion_started and line.strip():
                        if line.strip().startswith(('1.', '2.', '3.', '4.')):
                            suggestion_count += 1
                            # Afficher seulement les 2 premières suggestions
                            if suggestion_count <= 2:
                                print(f"   {line.strip()}")
                        elif line.startswith('**') and 'sources' in line.lower():
                            break
                
                print(f"\n📊 Nombre de suggestions détectées : {suggestion_count}")
                
            else:
                print("⚠️ Aucune suggestion détectée dans la réponse")
                print("🔍 Vérifiez que les prompts sont correctement configurés")
            
            # Afficher les 200 premiers caractères de la réponse
            print(f"\n📝 Aperçu de la réponse :")
            print("-" * 40)
            print(response[:400] + "..." if len(response) > 400 else response)
            print("-" * 40)
            
        else:
            print("❌ Aucune réponse générée")
            print("🔧 Vérifiez votre configuration et vos clés API")
        
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        print("💡 Vérifiez que vous avez bien remplacé le fichier src/simple_rag/graph.py")
        
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        print("🔧 Vérifiez votre configuration et vos clés API")
        import traceback
        traceback.print_exc()

async def test_fallback_suggestions():
    """Test des suggestions de fallback."""
    
    print("\n🔄 TEST DES SUGGESTIONS DE FALLBACK")
    print("=" * 40)
    
    try:
        from simple_rag.graph import generate_fallback_suggestions
        
        test_questions = [
            "population Sénégal",
            "pauvreté au Sénégal", 
            "emploi des jeunes",
            "question générale"
        ]
        
        for question in test_questions:
            print(f"\n📝 Question : {question}")
            suggestions = generate_fallback_suggestions(question)
            
            # Compter les suggestions
            suggestion_count = suggestions.count('?')
            print(f"✅ {suggestion_count} suggestions générées")
            
    except Exception as e:
        print(f"❌ Erreur test fallback : {e}")

if __name__ == "__main__":
    print("🇸🇳 TESTEUR D'INTÉGRATION ANSD")
    print("=" * 80)
    print("📄 Chargement de la configuration .env...")
    
    try:
        # Test principal
        asyncio.run(test_suggestions_integration())
        
        # Test des fonctions auxiliaires
        asyncio.run(test_fallback_suggestions())
        
        print("\n" + "=" * 80)
        print("✅ TESTS TERMINÉS")
        print("💡 Si vous voyez les suggestions, l'intégration est réussie !")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        print("\n🔧 Conseils de dépannage :")
        print("   • Vérifiez vos clés API dans .env")
        print("   • Vérifiez que tous les modules sont installés")
        print("   • Assurez-vous d'avoir remplacé le bon fichier graph.py")