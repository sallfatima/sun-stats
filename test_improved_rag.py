#!/usr/bin/env python3
"""
Script de test pour valider les améliorations du système RAG ANSD.
Version 2 avec chargement automatique du fichier .env
"""

# =============================================================================
# CHARGEMENT DU FICHIER .env (PREMIÈRE CHOSE À FAIRE)
# =============================================================================
from dotenv import load_dotenv
load_dotenv()
print("✅ Fichier .env chargé")

import asyncio
import time
import sys
import os

# Ajouter le répertoire src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Vérification des clés API après chargement .env
def verify_api_keys():
    """Vérifie que les clés API essentielles sont chargées."""
    print("\n🔍 Vérification des clés API:")
    
    api_keys = {
        'OPENAI_API_KEY': ('OpenAI', True),  # (description, required)
        'ANTHROPIC_API_KEY': ('Anthropic Claude', False),
        'PINECONE_API_KEY': ('Pinecone Vector DB', False),
        'LANGSMITH_API_KEY': ('LangSmith Tracing', False)
    }
    
    missing_required = []
    
    for key, (description, required) in api_keys.items():
        value = os.getenv(key)
        if value and value.strip():
            # Masquer les clés sensibles
            masked_value = value[:10] + '...' if len(value) > 10 else '***'
            print(f"   ✅ {description}: {masked_value}")
        else:
            if required:
                print(f"   ❌ {description}: MANQUANTE (REQUIS)")
                missing_required.append(key)
            else:
                print(f"   ⚪ {description}: Non configurée (optionnel)")
    
    if missing_required:
        print(f"\n🚨 Erreur: Clés requises manquantes: {', '.join(missing_required)}")
        print("💡 Vérifiez votre fichier .env")
        return False
    
    print("✅ Toutes les clés requises sont configurées")
    return True

# Vérifier les clés API
if not verify_api_keys():
    print("\n⏹️  Arrêt du script - Configuration API incomplète")
    sys.exit(1)

# Imports du système RAG (après vérification des clés)
try:
    from langchain_core.messages import HumanMessage
    print("✅ langchain_core importé")
    
    # Import direct du module sans passer par __init__.py pour éviter les imports circulaires
    try:
        from simple_rag.graph import graph
        from simple_rag.configuration import RagConfiguration
        print("✅ Modules RAG importés")
    except ImportError as e:
        print(f"⚠️  Import direct échoué: {e}")
        # Essayer import alternatif
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simple_rag'))
        from graph import graph
        from configuration import RagConfiguration
        print("✅ Modules RAG importés (méthode alternative)")

except ImportError as e:
    print(f"❌ Erreur d'import critique: {e}")
    print("💡 Vérifiez que toutes les dépendances sont installées:")
    print("   pip install langchain-core langgraph python-dotenv")
    sys.exit(1)

# Fonction de validation des réponses (basique, sans dépendances externes)
def validate_ansd_response(response: str) -> dict:
    """Valide qu'une réponse étendue contient les éléments requis pour l'ANSD."""
    
    validation = {
        'has_numerical_data': False,
        'has_source_citation': False,
        'has_year_reference': False,
        'has_ansd_terminology': False,
        'has_structure': False,
        'is_comprehensive': False,
        'has_external_ansd_knowledge': False,
        'quality_score': 0.0,
        'suggestions': []
    }
    
    response_lower = response.lower()
    
    # Vérifier la présence de données numériques
    import re
    if re.search(r'\d+(?:[.,]\d+)?(?:\s*%|\s*millions?|\s*milliards?|\s*habitants?)', response):
        validation['has_numerical_data'] = True
        validation['quality_score'] += 15
    else:
        validation['suggestions'].append("Ajouter des données chiffrées précises")
    
    # Vérifier les citations de sources ANSD (documents + publications)
    source_indicators = [
        'source :', 'page ', 'document', 'rgph', 'eds', 'esps', 'ehcvm', 'enes', 
        'ansd', 'recensement', 'enquête', 'rapport', 'publication'
    ]
    if any(term in response_lower for term in source_indicators):
        validation['has_source_citation'] = True
        validation['quality_score'] += 15
    else:
        validation['suggestions'].append("Citer les sources ANSD spécifiques")
    
    # Vérifier les références temporelles
    if re.search(r'20\d{2}|année\s+de\s+référence', response):
        validation['has_year_reference'] = True
        validation['quality_score'] += 15
    else:
        validation['suggestions'].append("Préciser l'année de référence des données")
    
    # Vérifier la terminologie ANSD
    ansd_terms = ['statistique', 'démographique', 'sénégal', 'méthodologie', 'indicateur']
    if any(term in response_lower for term in ansd_terms):
        validation['has_ansd_terminology'] = True
        validation['quality_score'] += 10
    else:
        validation['suggestions'].append("Utiliser la terminologie statistique appropriée")
    
    # Vérifier la structure de la réponse
    structure_markers = ['**réponse directe**', '**données précises**', '**contexte additionnel**', '-']
    if any(marker in response_lower for marker in structure_markers):
        validation['has_structure'] = True
        validation['quality_score'] += 15
    else:
        validation['suggestions'].append("Améliorer la structure de la réponse")
    
    # Vérifier le caractère complet/développé
    if len(response) > 500:  # Réponse développée
        validation['is_comprehensive'] = True
        validation['quality_score'] += 15
    else:
        validation['suggestions'].append("Développer davantage la réponse")
    
    # Vérifier l'utilisation de connaissances ANSD externes
    external_indicators = [
        'selon les publications ansd', 'd\'après les rapports ansd', 
        'site officiel ansd', 'www.ansd.sn', 'publications officielles'
    ]
    if any(indicator in response_lower for indicator in external_indicators):
        validation['has_external_ansd_knowledge'] = True
        validation['quality_score'] += 15
    else:
        # Pas de pénalité si pas de connaissances externes, mais bonus si présentes
        pass
    
    return validation

class ANSDRAGTester:
    """Classe pour tester le système RAG amélioré de l'ANSD."""
    
    def __init__(self):
        try:
            self.config = RagConfiguration()
            print("✅ Configuration RAG initialisée")
        except Exception as e:
            print(f"⚠️  Erreur configuration RAG: {e}")
            # Configuration par défaut minimale
            self.config = type('Config', (), {
                'model': 'openai/gpt-4o',
                'retrieval_k': 15,
                'enable_debug_logs': True
            })()
            print("✅ Configuration par défaut utilisée")
        
        self.test_questions = [
            "Quelle est la population du Sénégal selon le dernier RGPH ?",
            "Quel est le taux de pauvreté au Sénégal ?",
            "Qu'est-ce que le recensement de la population ?",
            "Quels sont les indicateurs de santé maternelle au Sénégal ?",
            "Comment évolue le taux d'alphabétisation au Sénégal ?"
        ]
    
    async def test_single_question(self, question: str, verbose: bool = True):
        """Teste une question unique et retourne les résultats."""
        
        if verbose:
            print(f"\n🔍 TEST: {question}")
            print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Préparer l'état initial
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Configuration pour le graphe
            if hasattr(self.config, '__dict__'):
                config_dict = {"configurable": self.config.__dict__}
            else:
                config_dict = {"configurable": self.config}
            
            # Exécuter le graphe RAG
            result = await graph.ainvoke(initial_state, config=config_dict)
            
            processing_time = time.time() - start_time
            
            # Extraire la réponse
            if result and "messages" in result and result["messages"]:
                response = result["messages"][-1].content
                documents = result.get("documents", [])
                
                if verbose:
                    print(f"✅ RÉPONSE GÉNÉRÉE ({processing_time:.2f}s):")
                    print("-" * 60)
                    print(response)
                    print("-" * 60)
                    print(f"📚 Documents utilisés: {len(documents)}")
                    
                    # Validation de la qualité
                    validation = validate_ansd_response(response)
                    print(f"\n📊 SCORE DE QUALITÉ: {validation['quality_score']}/100")
                    
                    # Détail des vérifications
                    checks = {
                        "Données numériques": validation['has_numerical_data'],
                        "Sources ANSD citées": validation['has_source_citation'],
                        "Année de référence": validation['has_year_reference'],
                        "Terminologie ANSD": validation['has_ansd_terminology'],
                        "Structure claire": validation['has_structure']
                    }
                    
                    print("🔍 Vérifications détaillées:")
                    for check, passed in checks.items():
                        status = "✅" if passed else "❌"
                        print(f"   {status} {check}")
                    
                    if validation['suggestions']:
                        print("\n💡 SUGGESTIONS D'AMÉLIORATION:")
                        for suggestion in validation['suggestions']:
                            print(f"   • {suggestion}")
                    
                    # Évaluation globale
                    if validation['quality_score'] >= 80:
                        print("\n🎉 EXCELLENT - Réponse de haute qualité !")
                    elif validation['quality_score'] >= 60:
                        print("\n👍 BON - Réponse satisfaisante")
                    elif validation['quality_score'] >= 40:
                        print("\n⚠️  MOYEN - Réponse à améliorer")
                    else:
                        print("\n❌ FAIBLE - Réponse nécessite des améliorations")
                
                return {
                    "question": question,
                    "response": response,
                    "processing_time": processing_time,
                    "document_count": len(documents),
                    "validation": validation,
                    "success": True
                }
            
            else:
                if verbose:
                    print("❌ ERREUR: Aucune réponse générée")
                return {
                    "question": question,
                    "response": None,
                    "processing_time": processing_time,
                    "success": False,
                    "error": "Aucune réponse générée"
                }
        
        except Exception as e:
            processing_time = time.time() - start_time
            if verbose:
                print(f"❌ ERREUR ({processing_time:.2f}s): {str(e)}")
                print(f"🔍 Type d'erreur: {type(e).__name__}")
                
                # Suggestions selon le type d'erreur
                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str:
                    print("💡 Suggestion: Limite de taux API atteinte - attendez et réessayez")
                elif "auth" in error_str or "api_key" in error_str:
                    print("💡 Suggestion: Problème d'authentification - vérifiez votre clé API")
                elif "connection" in error_str or "network" in error_str:
                    print("💡 Suggestion: Problème de connexion réseau")
                elif "not found" in error_str:
                    print("💡 Suggestion: Vérifiez que vos documents sont indexés")
                else:
                    print("💡 Suggestion: Vérifiez la configuration et les logs ci-dessus")
            
            return {
                "question": question,
                "response": None,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
    
    async def run_all_tests(self):
        """Exécute tous les tests et génère un rapport."""
        
        print("🚀 DÉBUT DE LA SUITE COMPLÈTE DE TESTS")
        print("=" * 80)
        
        results = []
        total_start_time = time.time()
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 TEST {i}/{len(self.test_questions)}")
            result = await self.test_single_question(question, verbose=True)
            results.append(result)
            
            # Pause entre les tests pour éviter les limites de taux
            if i < len(self.test_questions):
                print("\n⏸️  Pause de 2 secondes...")
                await asyncio.sleep(2)
        
        total_time = time.time() - total_start_time
        
        # Générer le rapport final
        self.generate_report(results, total_time)
        
        return results
    
    def generate_report(self, results, total_time):
        """Génère un rapport de synthèse des tests."""
        
        print("\n" + "=" * 80)
        print("📊 RAPPORT DE SYNTHÈSE - RAG ANSD AMÉLIORÉ")
        print("=" * 80)
        
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        print(f"✅ Tests réussis: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.0f}%)")
        print(f"❌ Tests échoués: {len(failed_tests)}/{len(results)}")
        print(f"⏱️  Temps total: {total_time:.2f}s")
        print(f"⏱️  Temps moyen par test: {total_time/len(results):.2f}s")
        
        if successful_tests:
            # Statistiques de qualité
            quality_scores = [r['validation']['quality_score'] for r in successful_tests]
            avg_quality = sum(quality_scores) / len(quality_scores)
            max_quality = max(quality_scores)
            min_quality = min(quality_scores)
            
            avg_docs = sum(r['document_count'] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
            
            print(f"\n📈 MÉTRIQUES DE QUALITÉ:")
            print(f"   📊 Score moyen: {avg_quality:.1f}/100")
            print(f"   🏆 Meilleur score: {max_quality:.0f}/100")
            print(f"   📉 Score le plus bas: {min_quality:.0f}/100")
            print(f"   📚 Documents utilisés en moyenne: {avg_docs:.1f}")
            print(f"   ⚡ Temps de réponse moyen: {avg_time:.2f}s")
            
            # Classification des résultats
            excellent = len([s for s in quality_scores if s >= 80])
            good = len([s for s in quality_scores if 60 <= s < 80])
            average = len([s for s in quality_scores if 40 <= s < 60])
            poor = len([s for s in quality_scores if s < 40])
            
            print(f"\n🎯 RÉPARTITION DES PERFORMANCES:")
            print(f"   🎉 Excellent (80-100): {excellent}")
            print(f"   👍 Bon (60-79): {good}")
            print(f"   ⚠️  Moyen (40-59): {average}")
            print(f"   ❌ Faible (<40): {poor}")
        
        # Détailler les échecs
        if failed_tests:
            print(f"\n❌ DÉTAIL DES ÉCHECS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"   {i}. {test['question']}")
                print(f"      💥 Erreur: {test['error']}")
        
        # Top 3 des meilleures réponses
        if successful_tests and len(successful_tests) >= 3:
            best_tests = sorted(successful_tests, key=lambda x: x['validation']['quality_score'], reverse=True)[:3]
            print(f"\n🏆 TOP 3 DES MEILLEURES RÉPONSES:")
            for i, test in enumerate(best_tests, 1):
                score = test['validation']['quality_score']
                time_taken = test['processing_time']
                print(f"   {i}. {test['question']}")
                print(f"      📊 Score: {score}/100 | ⏱️ {time_taken:.2f}s")
        
        # Recommandations finales
        print(f"\n💡 RECOMMANDATIONS:")
        if avg_quality >= 75:
            print("   🎉 Excellent ! Votre système RAG ANSD fonctionne parfaitement.")
            print("   ✅ Prêt pour la production")
        elif avg_quality >= 60:
            print("   👍 Bon fonctionnement avec quelques améliorations possibles.")
            print("   🔧 Affinez les prompts et la récupération de documents")
        elif avg_quality >= 40:
            print("   ⚠️  Performances moyennes - améliorations nécessaires.")
            print("   🔧 Vérifiez la qualité des documents indexés")
            print("   🔧 Améliorez les prompts système")
        else:
            print("   ❌ Performances faibles - révision complète recommandée.")
            print("   🔧 Vérifiez la configuration et les documents")
            print("   🔧 Consultez la documentation technique")
        
        print("=" * 80)

async def quick_test():
    """Test rapide avec une question simple."""
    
    tester = ANSDRAGTester()
    result = await tester.test_single_question(
        "Quelle est la population du Sénégal selon le RGPH ?",
        verbose=True
    )
    return result

async def performance_test():
    """Test de performance avec chronométrage détaillé."""
    
    print("⚡ TEST DE PERFORMANCE")
    print("=" * 50)
    
    tester = ANSDRAGTester()
    questions = [
        "Quelle est la population totale du Sénégal ?",
        "Quel est le taux de pauvreté ?",
        "Quels sont les indicateurs démographiques ?"
    ]
    
    times = []
    scores = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Test {i}/{len(questions)}: {question[:50]}...")
        start = time.time()
        result = await tester.test_single_question(question, verbose=False)
        end = time.time()
        
        processing_time = end - start
        times.append(processing_time)
        
        if result['success']:
            score = result['validation']['quality_score']
            scores.append(score)
            status = "✅"
            print(f"{status} Réussi en {processing_time:.2f}s (Qualité: {score}/100)")
        else:
            status = "❌"
            print(f"{status} Échoué en {processing_time:.2f}s - {result.get('error', 'Erreur inconnue')}")
    
    print(f"\n📊 RÉSULTATS DE PERFORMANCE:")
    print(f"   ⏱️  Temps moyen: {sum(times)/len(times):.2f}s")
    print(f"   ⏱️  Temps total: {sum(times):.2f}s")
    print(f"   🚀 Test le plus rapide: {min(times):.2f}s")
    print(f"   🐌 Test le plus lent: {max(times):.2f}s")
    
    if scores:
        print(f"   📊 Qualité moyenne: {sum(scores)/len(scores):.1f}/100")

async def test_specific_features():
    """Teste des fonctionnalités spécifiques du système amélioré."""
    
    print("🔧 TEST DES FONCTIONNALITÉS SPÉCIFIQUES")
    print("=" * 60)
    
    # Test 1: Vérifier que les améliorations sont actives
    print("\n1️⃣ Vérification des améliorations ANSD:")
    
    # Vérifier les imports des nouvelles fonctions
    try:
        from simple_rag.graph import preprocess_query_enhanced, format_docs_with_rich_metadata
        print("   ✅ Fonctions améliorées importées")
        
        # Test du prétraitement
        test_query = "population Sénégal"
        enhanced_query = preprocess_query_enhanced(test_query)
        print(f"   ✅ Prétraitement: '{test_query}' → '{enhanced_query}'")
        
    except ImportError:
        print("   ❌ Fonctions améliorées non disponibles")
        print("   💡 Vérifiez que vous avez bien remplacé src/simple_rag/graph.py")
    
    # Test 2: Configuration améliorée
    print("\n2️⃣ Test de la configuration améliorée:")
    try:
        from simple_rag.configuration import RagConfiguration
        config = RagConfiguration()
        
        enhanced_features = [
            'enable_query_preprocessing',
            'enable_document_scoring',
            'ansd_survey_weights'
        ]
        
        for feature in enhanced_features:
            if hasattr(config, feature):
                print(f"   ✅ {feature}: configuré")
            else:
                print(f"   ❌ {feature}: manquant")
                
    except Exception as e:
        print(f"   ❌ Erreur configuration: {e}")
    
    # Test 3: Test avec question spécifique ANSD
    print("\n3️⃣ Test avec terminologie ANSD spécifique:")
    tester = ANSDRAGTester()
    
    ansd_questions = [
        "Résultats du RGPH-5 sur la population",
        "Données de l'EDS sur la santé maternelle",
        "Indicateurs ESPS sur la pauvreté"
    ]
    
    for question in ansd_questions:
        print(f"\n   🔍 Test: {question}")
        result = await tester.test_single_question(question, verbose=False)
        
        if result['success']:
            score = result['validation']['quality_score']
            response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
            print(f"   ✅ Score: {score}/100")
            print(f"   📝 Aperçu: {response_preview}")
        else:
            print(f"   ❌ Échec: {result.get('error', 'Erreur inconnue')}")
    
    print("=" * 60)

async def main():
    """Fonction principale pour exécuter les tests."""
    
    print("🇸🇳 TESTEUR RAG ANSD - SYSTÈME AMÉLIORÉ v2")
    print("=" * 80)
    print("📄 Configuration .env chargée avec python-dotenv")
    print("🔧 Version avec nouvelles fonctionnalités activées")
    print("=" * 80)
    
    # Menu des options
    print("\nOptions de test disponibles:")
    print("1. Test rapide (1 question)")
    print("2. Test de performance (3 questions, focus vitesse)")
    print("3. Test de fonctionnalités spécifiques")
    print("4. Suite complète de tests (5 questions)")
    print("5. Quitter")
    
    try:
        choice = input("\nChoisissez une option (1-5): ").strip()
        
        if choice == "1":
            await quick_test()
        elif choice == "2":
            await performance_test()
        elif choice == "3":
            await test_specific_features()
        elif choice == "4":
            tester = ANSDRAGTester()
            await tester.run_all_tests()
        elif choice == "5":
            print("👋 Au revoir !")
            return
        else:
            print("❌ Option invalide. Exécution du test rapide par défaut.")
            await quick_test()
    
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrompus par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution des tests: {e}")
        print("🔧 Suggestions de dépannage:")
        print("   • Vérifiez vos clés API dans le fichier .env")
        print("   • Vérifiez que tous les modules sont installés")
        print("   • Consultez les logs d'erreur ci-dessus")

if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())