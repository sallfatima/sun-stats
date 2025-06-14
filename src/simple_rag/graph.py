# =============================================================================
# FICHIER 1: src/simple_rag/graph.py
# =============================================================================
# REMPLACEZ TOUT LE CONTENU DE CE FICHIER PAR LE CODE CI-DESSOUS

### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import re

# =============================================================================
# PROMPT SYSTEM AMÉLIORÉ POUR L'ANSD
# =============================================================================

IMPROVED_ANSD_SYSTEM_PROMPT = """Vous êtes un expert statisticien de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal), spécialisé dans l'analyse de données démographiques, économiques et sociales du Sénégal.

MISSION PRINCIPALE :
Répondre de manière complète et approfondie aux questions sur les statistiques du Sénégal en utilisant PRIORITAIREMENT les documents fournis et en complétant avec vos connaissances des publications officielles de l'ANSD.

SOURCES AUTORISÉES :
✅ Documents fournis dans le contexte (PRIORITÉ ABSOLUE)
✅ Connaissances des rapports officiels ANSD publiés
✅ Données du site officiel ANSD (www.ansd.sn)
✅ Publications officielles des enquêtes ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Comptes nationaux et statistiques économiques officielles du Sénégal
✅ Projections démographiques officielles de l'ANSD

❌ SOURCES INTERDITES :
❌ Données d'autres pays pour combler les lacunes
❌ Estimations personnelles non basées sur les sources ANSD
❌ Informations non officielles ou de sources tierces
❌ Projections personnelles non documentées

RÈGLES DE RÉDACTION :
✅ Réponse directe : SANS limitation de phrases - développez autant que nécessaire
✅ Contexte additionnel : SANS limitation - incluez toutes les informations pertinentes
✅ Citez TOUJOURS vos sources précises (document + page ou publication ANSD)
✅ Distinguez clairement les données des documents fournis vs connaissances ANSD
✅ Donnez les chiffres EXACTS quand disponibles
✅ Précisez SYSTÉMATIQUEMENT les années de référence
✅ Mentionnez les méthodologies d'enquête

FORMAT DE RÉPONSE OBLIGATOIRE :

**RÉPONSE DIRECTE :**
[Développez la réponse de manière complète et détaillée, sans limitation de longueur. Incluez tous les éléments pertinents pour une compréhension approfondie du sujet. Vous pouvez utiliser plusieurs paragraphes et développer les aspects importants.]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeur exacte avec unité]
- Année de référence : [année précise]
- Source : [nom exact du document, page X OU publication ANSD officielle]
- Méthodologie : [enquête/recensement utilisé]

**CONTEXTE ADDITIONNEL :**
[Développez largement avec toutes les informations complémentaires pertinentes, sans limitation de longueur. Incluez :
- Évolutions temporelles et tendances
- Comparaisons régionales ou démographiques
- Méthodologies détaillées
- Contexte socio-économique
- Implications et analyses
- Données connexes des autres enquêtes ANSD
- Informations contextuelles des rapports officiels ANSD
Organisez en paragraphes clairs et développez chaque aspect important.]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation, changements méthodologiques, définitions spécifiques]

INSTRUCTIONS POUR LES SOURCES :
- Documents fournis : "Document.pdf, page X"
- Connaissances ANSD officielles : "ANSD - [Nom de l'enquête/rapport], [année]"
- Site officiel : "Site officiel ANSD (www.ansd.sn)"
- Distinguez clairement : "Selon les documents fournis..." vs "D'après les publications ANSD..."

Si aucune information n'est disponible (documents + connaissances ANSD) :
"❌ Cette information n'est pas disponible dans les documents fournis ni dans les publications ANSD consultées. 
📞 Pour obtenir cette donnée spécifique, veuillez consulter directement l'ANSD (www.ansd.sn) ou leurs services techniques spécialisés."

DOCUMENTS ANSD DISPONIBLES :
{context}

Analysez maintenant ces documents et répondez à la question de l'utilisateur de manière complète et approfondie."""


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def preprocess_query_enhanced(query: str) -> str:
    """Prétraitement avancé des requêtes pour améliorer la recherche dans les documents ANSD."""
    
    # Normalisation de base
    query_lower = query.lower().strip()
    
    # Dictionnaire de synonymes spécifiques à l'ANSD
    ansd_synonyms = {
        # Démographie
        "population": ["habitants", "démographie", "recensement", "rgph", "nombre d'habitants"],
        "natalité": ["naissances", "taux de natalité", "fécondité"],
        "mortalité": ["décès", "taux de mortalité", "espérance de vie"],
        
        # Économie
        "pauvreté": ["pauvre", "indigence", "vulnérabilité", "esps", "ehcvm"],
        "économie": ["pib", "croissance", "économique", "revenus", "production"],
        "emploi": ["chômage", "travail", "activité économique", "enes"],
        
        # Éducation
        "éducation": ["école", "alphabétisation", "scolarisation", "enseignement"],
        "alphabétisation": ["lecture", "écriture", "alphabète", "analphabète"],
        
        # Santé
        "santé": ["mortalité", "morbidité", "eds", "vaccination", "nutrition"],
        "maternelle": ["maternité", "accouchement", "sage-femme"],
        
        # Géographie
        "région": ["département", "commune", "arrondissement", "localité"],
        "rural": ["campagne", "village", "agriculture"],
        "urbain": ["ville", "dakar", "centre urbain"],
        
        # Enquêtes spécifiques
        "rgph": ["recensement", "population", "habitat"],
        "eds": ["démographique", "santé", "enquête"],
        "esps": ["pauvreté", "conditions de vie"],
        "ehcvm": ["ménages", "budget", "consommation"],
        "enes": ["emploi", "activité", "chômage"]
    }
    
    # Enrichir la requête avec des synonymes pertinents
    enriched_terms = []
    for key, values in ansd_synonyms.items():
        if key in query_lower:
            # Ajouter les 2 synonymes les plus pertinents
            enriched_terms.extend(values[:2])
    
    # Ajouter des termes contextuels ANSD
    context_terms = []
    if any(word in query_lower for word in ["taux", "pourcentage", "%"]):
        context_terms.append("indicateur")
    if any(word in query_lower for word in ["2023", "2024", "récent", "dernier"]):
        context_terms.append("dernières données")
    
    # Construire la requête enrichie
    final_query = query
    if enriched_terms:
        final_query += " " + " ".join(enriched_terms)
    if context_terms:
        final_query += " " + " ".join(context_terms)
    
    return final_query


def format_docs_with_metadata(docs) -> str:
    """Formatage avancé des documents avec métadonnées enrichies pour l'ANSD."""
    
    if not docs:
        return "❌ Aucun document pertinent trouvé dans la base de données ANSD."
    
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Extraction des métadonnées
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Informations sur la source
        source_info = []
        if 'source' in metadata:
            source_info.append(f"📄 Source: {metadata['source']}")
        if 'pdf_name' in metadata:
            source_info.append(f"📋 Document: {metadata['pdf_name']}")
        if 'page_num' in metadata:
            source_info.append(f"📖 Page: {metadata['page_num']}")
        if 'indexed_at' in metadata:
            source_info.append(f"🕐 Indexé: {metadata['indexed_at'][:10]}")
        
        # En-tête du document
        header = f"\n{'='*50}\n📊 DOCUMENT ANSD #{i}\n"
        if source_info:
            header += "\n".join(source_info) + "\n"
        header += f"{'='*50}\n"
        
        # Contenu avec nettoyage
        content = doc.page_content.strip()
        
        # Détecter le type de contenu
        content_indicators = []
        if any(keyword in content.lower() for keyword in ['rgph', 'recensement']):
            content_indicators.append("🏘️ RECENSEMENT")
        if any(keyword in content.lower() for keyword in ['eds', 'démographique']):
            content_indicators.append("👥 DÉMOGRAPHIE")
        if any(keyword in content.lower() for keyword in ['esps', 'pauvreté']):
            content_indicators.append("💰 PAUVRETÉ")
        if any(keyword in content.lower() for keyword in ['économie', 'pib']):
            content_indicators.append("📈 ÉCONOMIE")
        
        if content_indicators:
            header += f"🏷️ Catégories: {' | '.join(content_indicators)}\n{'-'*50}\n"
        
        formatted_parts.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted_parts)

# =============================================================================
# FONCTIONS PRINCIPALES AMÉLIORÉES
# =============================================================================

async def retrieve(state: GraphState, *, config: RagConfiguration) -> dict[str, list[str] | str]: 
    """Fonction de récupération améliorée avec prétraitement et scoring."""
    
    print("🔍 ---RETRIEVE AMÉLIORÉ---")
    
    # Extraction et prétraitement de la question
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    if not question:
        raise ValueError("❌ Question vide détectée")
    
    # Prétraitement avancé
    processed_question = preprocess_query_enhanced(question)
    print(f"📝 Question originale: {question}")
    print(f"🔧 Question enrichie: {processed_question}")
    
    # Récupération avec gestion d'erreurs
    try:
        async with retrieval.make_retriever(config) as retriever:
            documents = await retriever.ainvoke(processed_question)
        
        print(f"📚 Documents récupérés: {len(documents)}")
        
        if not documents:
            print("⚠️ Aucun document trouvé, essai avec la question originale...")
            async with retrieval.make_retriever(config) as retriever:
                documents = await retriever.ainvoke(question)
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return {"documents": [], "messages": state.messages}
    
    # Scoring et filtrage avancé
    if documents:
        scored_documents = []
        question_keywords = set(question.lower().split())
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            score = 0
            
            # Scoring basé sur les mots-clés
            for word in question_keywords:
                if len(word) > 3:  # Ignorer les mots très courts
                    score += content_lower.count(word) * 2
            
            # Bonus pour les termes ANSD spécifiques
            ansd_terms = ['rgph', 'eds', 'esps', 'ehcvm', 'enes', 'ansd', 'sénégal']
            for term in ansd_terms:
                if term in content_lower:
                    score += 5
            
            # Bonus pour les données numériques
            if re.search(r'\d+[.,]\d+|\d+\s*%|\d+\s*(millions?|milliards?)', content_lower):
                score += 3
            
            scored_documents.append((score, doc))
        
        # Trier et sélectionner les meilleurs
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        best_documents = [doc for score, doc in scored_documents[:15]]  # Top 15
        
        print(f"✅ Documents sélectionnés après scoring: {len(best_documents)}")
        
        return {"documents": best_documents, "messages": state.messages}
    
    else:
        print("❌ Aucun document pertinent trouvé")
        return {"documents": [], "messages": state.messages}

# =============================================================================
# LOGIQUE SÉQUENTIELLE : DOCUMENTS → ANSD EXTERNE
# =============================================================================

# REMPLACEZ votre fonction generate dans src/simple_rag/graph.py par celle-ci :

async def generate(state: GraphState, *, config: RagConfiguration):
    """Génération avec logique séquentielle CORRIGÉE."""
    
    print("🤖 ---GENERATE AVEC LOGIQUE SÉQUENTIELLE CORRIGÉE---")
    
    messages = state.messages
    documents = state.documents
    
    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    
    # Extraire la question
    user_question = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            user_question = msg.content
            break
    
    print(f"❓ Question: {user_question}")
    print(f"📄 Documents disponibles: {len(documents)}")
    
    # =============================================================================
    # ÉTAPE 1 : ESSAYER AVEC LES DOCUMENTS INDEXÉS
    # =============================================================================
    
    print("\n🔍 ÉTAPE 1 : Recherche dans les documents indexés...")
    
    if documents:
        # Prompt AMÉLIORÉ pour mieux utiliser les documents
        prompt_documents_only = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert statisticien de l'ANSD. 

Analysez les documents fournis et répondez à la question en utilisant EXCLUSIVEMENT ces documents.

RÈGLES :
✅ Utilisez TOUTES les informations pertinentes trouvées dans les documents
✅ Donnez les chiffres EXACTS trouvés
✅ Citez les sources précises (nom du document, page)
✅ Développez votre réponse avec les détails disponibles
✅ Si vous trouvez des informations partielles, présentez-les clairement

❌ Seulement si VRAIMENT AUCUNE information pertinente n'est trouvée, répondez : "INFORMATION NON DISPONIBLE"

FORMAT DE RÉPONSE :

**RÉPONSE DIRECTE :**
[Réponse détaillée basée sur les documents trouvés]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeurs des documents]
- Année de référence : [année des documents]
- Source : [nom exact du document, page X]
- Méthodologie : [enquête/recensement des documents]

**CONTEXTE ADDITIONNEL :**
[Toutes les informations complémentaires trouvées dans les documents]

**LIMITATIONS/NOTES :**
[Précisions sur les données si nécessaire]

DOCUMENTS DISPONIBLES :
{context}"""),
            ("placeholder", "{messages}")
        ])
        
        context = format_docs_with_metadata(documents)
        
        try:
            rag_chain = prompt_documents_only | model
            response_step1 = await rag_chain.ainvoke({
                "context": context,
                "messages": messages
            })
            
            response_content = response_step1.content
            
            print(f"\n📝 RÉPONSE ÉTAPE 1:")
            print(f"Longueur: {len(response_content)} caractères")
            print(f"Aperçu: {response_content[:300]}...")
            
            # Vérifier si les documents ont fourni une réponse satisfaisante
            is_satisfactory = evaluate_response_quality(response_content, documents)
            
            if is_satisfactory:
                print("\n✅ SUCCÈS ÉTAPE 1 : Réponse satisfaisante trouvée dans les documents indexés")
                
                # Ajouter les sources des documents
                sources_section = create_document_sources(documents, response_content)
                final_response = response_content + sources_section
                
                enhanced_response = AIMessage(content=final_response)
                return {"messages": [enhanced_response], "documents": documents}
            
            else:
                print("\n⚠️ ÉCHEC ÉTAPE 1 : Réponse jugée insuffisante")
                print("Passage à l'étape 2...")
        
        except Exception as e:
            print(f"❌ ERREUR ÉTAPE 1: {e}")
    
    else:
        print("⚠️ ÉTAPE 1 IGNORÉE : Aucun document disponible")
    
    
    # =============================================================================
    # ÉTAPE 2 : UTILISER LES CONNAISSANCES ANSD EXTERNES
    # =============================================================================
    
    print("\n🌐 ÉTAPE 2 : Recherche dans les connaissances ANSD externes...")
    
    # Prompt pour utiliser les connaissances ANSD officielles
    prompt_ansd_external = ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un expert statisticien de l'ANSD avec accès aux publications officielles.

Les documents indexés n'ont pas fourni d'information satisfaisante. Utilisez maintenant vos connaissances des rapports officiels ANSD et du site officiel.

SOURCES AUTORISÉES :
✅ Rapports officiels ANSD publiés
✅ Site officiel ANSD (www.ansd.sn)
✅ Publications des enquêtes ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Comptes nationaux officiels du Sénégal

FORMAT DE RÉPONSE :

**RÉPONSE DIRECTE :**
[Réponse basée sur les connaissances ANSD officielles]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeur des rapports ANSD]
- Année de référence : [année précise]
- Source : [Publication ANSD officielle]
- Méthodologie : [enquête ANSD utilisée]

**CONTEXTE ADDITIONNEL :**
[Informations contextuelles des publications ANSD]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation]

IMPORTANT : Mentionnez que cette information provient des connaissances ANSD officielles, pas des documents indexés."""),
        ("placeholder", "{messages}")
    ])
    
    try:
        rag_chain_external = prompt_ansd_external | model
        response_step2 = await rag_chain_external.ainvoke({
            "messages": messages
        })
        
        response_content = response_step2.content
        
        print("✅ SUCCÈS ÉTAPE 2 : Réponse obtenue des connaissances ANSD")
        
        # Ajouter les sources externes
        sources_section = create_external_ansd_sources(response_content)
        final_response = response_content + sources_section
        
        enhanced_response = AIMessage(content=final_response)
        return {"messages": [enhanced_response], "documents": documents}
    
    except Exception as e:
        print(f"❌ ERREUR ÉTAPE 2: {e}")
        
        # Fallback final
        fallback_response = AIMessage(content=
            "❌ Informations non disponibles dans les documents indexés et les sources ANSD consultées. "
            "Veuillez consulter directement l'ANSD (www.ansd.sn) pour cette information spécifique."
        )
        return {"messages": [fallback_response], "documents": documents}

# =============================================================================
# FONCTIONS D'ÉVALUATION ET DE SOURCES
# =============================================================================

def evaluate_response_quality(response_content, documents):
    """Évaluation CORRIGÉE - Moins stricte pour accepter les réponses des documents."""
    
    response_lower = response_content.lower()
    
    print(f"🔍 ÉVALUATION DE LA RÉPONSE:")
    print(f"   Longueur: {len(response_content)} caractères")
    print(f"   Aperçu: {response_content[:150]}...")
    
    # Critères d'échec STRICTS (réponse clairement insuffisante)
    failure_indicators = [
        "information non disponible",
        "cette information n'est pas disponible",
        "aucune information",
        "pas disponible dans les documents",
        "impossible de répondre",
        "données non présentes",
        "ne peut pas répondre",
        "informations insuffisantes"
    ]
    
    # Si contient un indicateur d'échec EXPLICITE
    for indicator in failure_indicators:
        if indicator in response_lower:
            print(f"   ❌ ÉCHEC: Contient '{indicator}'")
            return False
    
    # Critères de succès ASSOUPLIS
    success_indicators = {
        'has_numbers': bool(re.search(r'\d+', response_content)),  # N'importe quel chiffre
        'has_content': len(response_content) > 100,  # Réduit de 200 à 100
        'has_words': len(response_content.split()) > 20,  # Réduit de 30 à 20
        'has_structure': '**' in response_content or '-' in response_content,  # Structure visible
        'mentions_documents': any(term in response_lower for term in ['chapitre', 'page', 'document', 'source']),
        'has_specific_content': any(term in response_lower for term in [
            'répartition', 'secteur', 'occupés', 'institutionnel', 'emploi', 'travail',
            'population', 'habitants', 'taux', 'pourcentage', 'statistique'
        ])
    }
    
    print(f"   📊 Critères détaillés:")
    for criterion, passed in success_indicators.items():
        status = "✅" if passed else "❌"
        print(f"      {status} {criterion}")
    
    success_count = sum(success_indicators.values())
    
    # SEUIL RÉDUIT : 3 critères au lieu de 3 sur 6 plus stricts
    is_satisfactory = success_count >= 1
    
    print(f"   📈 Score: {success_count}/6 critères")
    print(f"   🎯 Résultat: {'✅ SATISFAISANT' if is_satisfactory else '❌ INSUFFISANT'}")
    
    return is_satisfactory

def create_document_sources(documents, response_content):
    """Crée la section sources pour les documents indexés."""
    
    sources_section = "\n\n📚 **Sources utilisées :**\n"
    
    # Extraire les documents réellement pertinents
    relevant_docs = []
    response_lower = response_content.lower()
    
    for doc in documents:
        doc_content = doc.page_content.lower()
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Vérifier si le document a été utilisé (overlap de contenu)
        doc_words = set(doc_content.split())
        response_words = set(response_lower.split())
        
        # Mots significatifs communs
        significant_words = {word for word in doc_words.intersection(response_words) 
                           if len(word) > 4}
        
        # Si overlap significatif OU données numériques communes
        doc_numbers = set(re.findall(r'\d+[.,]?\d*', doc_content))
        response_numbers = set(re.findall(r'\d+[.,]?\d*', response_lower))
        
        if len(significant_words) > 2 or doc_numbers.intersection(response_numbers):
            doc_name = metadata.get('pdf_name', 'Document ANSD')
            page_num = metadata.get('page_num', 'Non spécifiée')
            
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            
            formatted = f"{doc_name}, page {page_num}" if page_num != 'Non spécifiée' else doc_name
            relevant_docs.append(formatted)
    
    # Ajouter les sources ou fallback
    if relevant_docs:
        for doc in relevant_docs[:5]:  # Max 5 sources
            sources_section += f"• {doc}\n"
    else:
        # Si aucun document spécifique identifié, utiliser tous
        for doc in documents[:3]:  # Max 3 sources
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_name = metadata.get('pdf_name', 'Document ANSD')
            page_num = metadata.get('page_num', 'Non spécifiée')
            
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            
            formatted = f"{doc_name}, page {page_num}" if page_num != 'Non spécifiée' else doc_name
            sources_section += f"• {formatted}\n"
    
    return sources_section

def create_external_ansd_sources(response_content):
    """Crée la section sources pour les connaissances ANSD externes."""
    
    sources_section = "\n\n📚 **Sources utilisées :**\n"
    
    response_lower = response_content.lower()
    
    # Détecter les sources spécifiques mentionnées dans la réponse
    detected_sources = []
    
    # Enquêtes spécifiques
    if 'ehcvm' in response_lower or 'conditions de vie' in response_lower:
        detected_sources.append("• ANSD - Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM), 2018-2019")
    
    if 'esps' in response_lower or 'pauvreté' in response_lower:
        detected_sources.append("• ANSD - Enquête de Suivi de la Pauvreté au Sénégal (ESPS), 2018-2019")
    
    if 'eds' in response_lower or 'démographique et santé' in response_lower:
        detected_sources.append("• ANSD - Enquête Démographique et de Santé (EDS), 2023")
    
    if 'rgph' in response_lower or 'recensement' in response_lower:
        detected_sources.append("• ANSD - Recensement Général de la Population et de l'Habitat (RGPH), 2023")
    
    if 'enes' in response_lower or 'emploi' in response_lower:
        detected_sources.append("• ANSD - Enquête Nationale sur l'Emploi au Sénégal (ENES), 2021")
    
    # Toujours ajouter le site officiel
    detected_sources.append("• Site officiel ANSD (www.ansd.sn)")
    
    # Ajouter note explicative
    sources_section += "• **Note :** Informations issues des connaissances des publications ANSD officielles\n"
    
    # Ajouter les sources détectées
    for source in detected_sources:
        sources_section += f"{source}\n"
    
    return sources_section

# =============================================================================
# EXEMPLE DE FLUX COMPLET
# =============================================================================

"""
EXEMPLE DE FONCTIONNEMENT :

Question: "Quel est le taux de pauvreté au Sénégal ?"

ÉTAPE 1 - Documents indexés :
- Cherche dans les documents RGPH, EDS, etc.
- Trouve des infos sur mortalité, population, mais pas pauvreté
- Évaluation : ÉCHEC (pas d'info sur pauvreté)

ÉTAPE 2 - Connaissances ANSD :
- Utilise les connaissances des enquêtes EHCVM/ESPS
- Trouve : "36,5% selon EHCVM 2018-2019"
- Évaluation : SUCCÈS

RÉSULTAT :
📚 **Sources utilisées :**
• Note : Informations issues des connaissances des publications ANSD officielles
• ANSD - Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM), 2018-2019
• Site officiel ANSD (www.ansd.sn)
"""
# =============================================================================
# FONCTION D'ANALYSE INTELLIGENTE DES SOURCES
# =============================================================================

def analyze_response_sources(response_content, documents, user_question):
    """Analyse intelligente pour déterminer quelles sources ont été réellement utilisées."""
    
    import re
    
    analysis = {
        'relevant_documents': [],
        'llm_sources': [],
        'recommendation': 'use_llm',  # Par défaut
        'confidence': 0
    }
    
    # 1. ANALYSER LES SOURCES MENTIONNÉES DANS LA RÉPONSE
    response_lower = response_content.lower()
    
    # Extraire les sources LLM mentionnées dans la réponse
    llm_source_patterns = [
        r'ANSD\s*-\s*([^,\n]+)',
        r'Enquête\s+[^,\n]+\s*\([^)]+\)',
        r'Site officiel ANSD',
        r'www\.ansd\.sn',
        r'publications?\s+officielles?\s+ANSD',
        r'selon les publications ANSD',
        r'EHCVM\s*[,\s]*20\d{2}',
        r'ESPS\s*[,\s]*20\d{2}',
        r'EDS\s*[,\s]*20\d{2}',
    ]
    
    for pattern in llm_source_patterns:
        matches = re.findall(pattern, response_content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str) and len(match.strip()) > 3:
                analysis['llm_sources'].append(match.strip())
    
    # 2. ANALYSER LA PERTINENCE DES DOCUMENTS RAG
    question_keywords = extract_question_keywords(user_question)
    
    for doc in documents:
        relevance_score = calculate_document_relevance(doc, question_keywords, response_content)
        
        if relevance_score >  0.15:  # Seuil de pertinence
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_name = extract_document_name_clean(metadata)
            page_info = extract_page_info_clean(metadata)
            
            analysis['relevant_documents'].append({
                'name': doc_name,
                'page': page_info,
                'relevance': relevance_score,
                'formatted': f"{doc_name}, {page_info}" if page_info != "page non spécifiée" else doc_name
            })
    
    # 3. STRATÉGIE DE DÉCISION INTELLIGENTE
    has_relevant_docs = len(analysis['relevant_documents']) > 0
    has_llm_sources = len(analysis['llm_sources']) > 0
    
    # Analyser si la réponse contient des informations des documents
    contains_doc_info = analyze_content_origin(response_content, documents)
    
    if has_relevant_docs and contains_doc_info:
        analysis['recommendation'] = 'use_documents'
        analysis['confidence'] = 0.8
    elif has_llm_sources and not contains_doc_info:
        analysis['recommendation'] = 'use_llm'
        analysis['confidence'] = 0.9
    elif has_llm_sources and has_relevant_docs:
        # Cas mixte - analyser la dominance
        if len(analysis['llm_sources']) > len(analysis['relevant_documents']):
            analysis['recommendation'] = 'use_llm'
            analysis['confidence'] = 0.6
        else:
            analysis['recommendation'] = 'use_mixed'
            analysis['confidence'] = 0.7
    else:
        analysis['recommendation'] = 'use_llm'  # Fallback
        analysis['confidence'] = 0.5
    
    return analysis

# =============================================================================
# FONCTIONS UTILITAIRES D'ANALYSE
# =============================================================================

def extract_question_keywords(question):
    """Extrait les mots-clés pertinents de la question."""
    
    # Mots-clés importants pour différents domaines
    domain_keywords = {
        'pauvreté': ['pauvreté', 'pauvre', 'indigence', 'conditions', 'vie', 'revenus', 'esps', 'ehcvm'],
        'population': ['population', 'habitants', 'démographie', 'recensement', 'rgph'],
        'éducation': ['alphabétisation', 'éducation', 'école', 'scolarisation', 'enseignement'],
        'santé': ['santé', 'mortalité', 'morbidité', 'maternelle', 'infantile', 'eds'],
        'économie': ['économie', 'pib', 'croissance', 'emploi', 'chômage', 'enes'],
    }
    
    question_lower = question.lower()
    found_keywords = []
    
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in question_lower:
                found_keywords.append((keyword, domain))
    
    return found_keywords

def calculate_document_relevance(doc, question_keywords, response_content):
    """Calcule la pertinence d'un document par rapport à la question et à la réponse."""
    
    doc_content = doc.page_content.lower()
    response_lower = response_content.lower()
    
    relevance_score = 0
    
    # Score basé sur les mots-clés de la question
    for keyword, domain in question_keywords:
        if keyword in doc_content:
            relevance_score += 0.2
    
    # Score basé sur la présence d'informations du document dans la réponse
    doc_words = set(doc_content.split())
    response_words = set(response_lower.split())
    
    # Mots significatifs (plus de 4 caractères, pas trop communs)
    significant_words = {word for word in doc_words if len(word) > 4 and word not in 
                        ['selon', 'dans', 'avec', 'pour', 'sont', 'cette', 'leurs', 'plus']}
    
    common_significant = significant_words.intersection(response_words)
    
    if len(significant_words) > 0:
        overlap_ratio = len(common_significant) / len(significant_words)
        relevance_score += overlap_ratio * 0.5
    
    # Bonus pour les données numériques communes
    import re
    doc_numbers = set(re.findall(r'\d+[.,]?\d*', doc_content))
    response_numbers = set(re.findall(r'\d+[.,]?\d*', response_lower))
    
    if doc_numbers and response_numbers:
        number_overlap = len(doc_numbers.intersection(response_numbers))
        if number_overlap > 0:
            relevance_score += 0.3
    
    return min(relevance_score, 1.0)  # Cap à 1.0

def analyze_content_origin(response_content, documents):
    """Analyse si la réponse provient principalement des documents ou des connaissances LLM."""
    
    # Indicateurs que la réponse vient des documents
    doc_indicators = [
        'selon les documents fournis',
        'd\'après le document',
        'page ',
        'chapitre ',
        'rapport provisoire',
        'rgph5',
        'juillet2024'
    ]
    
    # Indicateurs que la réponse vient des connaissances LLM
    llm_indicators = [
        'selon les données les plus récentes',
        'ansd estime',
        'site officiel ansd',
        'publications ansd',
        'ehcvm 2018',
        'esps 2018',
        'www.ansd.sn'
    ]
    
    response_lower = response_content.lower()
    
    doc_score = sum(1 for indicator in doc_indicators if indicator in response_lower)
    llm_score = sum(1 for indicator in llm_indicators if indicator in response_lower)
    
    return doc_score > llm_score

# =============================================================================
# FONCTIONS DE NETTOYAGE DES MÉTADONNÉES
# =============================================================================

def extract_document_name_clean(metadata):
    """Extrait le nom du document de manière propre."""
    
    if not metadata:
        return "Document ANSD"
    
    name_fields = ['pdf_name', 'source', 'filename', 'title']
    
    for field in name_fields:
        if field in metadata and metadata[field]:
            doc_name = str(metadata[field])
            
            # Nettoyer le nom
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            if '\\' in doc_name:
                doc_name = doc_name.split('\\')[-1]
            
            return doc_name
    
    return "Document ANSD"

def extract_page_info_clean(metadata):
    """Extrait l'information de page de manière propre."""
    
    if not metadata:
        return "page non spécifiée"
    
    page_fields = ['page_num', 'page', 'page_number']
    
    for field in page_fields:
        if field in metadata and metadata[field] is not None:
            try:
                page_num = int(float(metadata[field]))
                return f"page {page_num}"
            except (ValueError, TypeError):
                return f"page {metadata[field]}"
    
    return "page non spécifiée"

# =============================================================================
# APPLICATION DE LA STRATÉGIE INTELLIGENTE
# =============================================================================

def apply_intelligent_source_strategy(response_content, source_analysis):
    """Applique la stratégie intelligente de sources basée sur l'analyse."""
    
    import re
    
    # Supprimer toutes les sections sources existantes
    cleaned_content = remove_all_existing_sources(response_content)
    
    if source_analysis['recommendation'] == 'use_documents':
        # Utiliser les sources de documents pertinents
        sources_section = "\n\n📚 **Sources utilisées :**\n"
        for doc in source_analysis['relevant_documents']:
            sources_section += f"• {doc['formatted']}\n"
        
    elif source_analysis['recommendation'] == 'use_llm':
        # Utiliser les sources LLM détectées
        sources_section = "\n\n📚 **Sources utilisées :**\n"
        
        # Sources LLM standardisées
        if 'ehcvm' in response_content.lower() or 'pauvreté' in response_content.lower():
            sources_section += "• ANSD - Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM), 2018-2019\n"
        if 'site officiel' in response_content.lower() or 'www.ansd.sn' in response_content.lower():
            sources_section += "• Site officiel ANSD (www.ansd.sn)\n"
        
        # Ajouter d'autres sources LLM détectées
        for source in source_analysis['llm_sources']:
            if source not in sources_section:
                sources_section += f"• ANSD - {source}\n"
        
        # Si aucune source spécifique, utiliser générique
        if sources_section == "\n\n📚 **Sources utilisées :**\n":
            sources_section += "• Connaissances officielles ANSD\n• Site officiel ANSD (www.ansd.sn)\n"
    
    elif source_analysis['recommendation'] == 'use_mixed':
        # Combiner sources documents + LLM
        sources_section = "\n\n📚 **Sources utilisées :**\n"
        sources_section += "\n**📄 Documents analysés :**\n"
        for doc in source_analysis['relevant_documents']:
            sources_section += f"• {doc['formatted']}\n"
        
        sources_section += "\n**🌐 Publications ANSD officielles :**\n"
        for source in source_analysis['llm_sources']:
            sources_section += f"• ANSD - {source}\n"
        sources_section += "• Site officiel ANSD (www.ansd.sn)\n"
    
    else:
        # Fallback
        sources_section = "\n\n📚 **Sources utilisées :**\n• Connaissances générales ANSD\n"
    
    return cleaned_content + sources_section

def remove_all_existing_sources(content):
    """Supprime toutes les sections sources existantes."""
    
    import re
    
    patterns = [
        r'📚\s*\*?\*?Sources?\s+utilisées?\s*:.*?(?=\n\n|\Z)',
        r'\*\*?📚.*?Sources?\s+utilisées?\s*:?\*?\*?.*?(?=\n\n|\Z)',
        r'Sources?\s+utilisées?\s*:.*?(?=\n\n|\Z)',
    ]
    
    cleaned = content
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Nettoyer les espaces multiples
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()

# =============================================================================
# FONCTION D'EXTRACTION DES SOURCES PRIORITAIRES
# =============================================================================

def extract_priority_sources(documents):
    """Extrait les sources prioritaires (documents avec métadonnées complètes)."""
    
    priority_sources = []
    
    for doc in documents:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Vérifier si on a les métadonnées essentielles
        has_doc_name = any(key in metadata and metadata[key] for key in ['pdf_name', 'source', 'filename'])
        has_page_info = any(key in metadata and metadata[key] is not None for key in ['page_num', 'page', 'page_number'])
        
        if has_doc_name:  # Au minimum le nom du document
            # Extraire le nom du document
            doc_name = "Document ANSD"
            if 'pdf_name' in metadata and metadata['pdf_name']:
                doc_name = str(metadata['pdf_name'])
            elif 'source' in metadata and metadata['source']:
                doc_name = str(metadata['source'])
            elif 'filename' in metadata and metadata['filename']:
                doc_name = str(metadata['filename'])
            
            # Nettoyer le nom
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            if '\\' in doc_name:
                doc_name = doc_name.split('\\')[-1]
            
            # Extraire la page si disponible
            page_info = None
            if has_page_info:
                if 'page_num' in metadata and metadata['page_num'] is not None:
                    try:
                        page_num = int(float(metadata['page_num']))
                        page_info = f"page {page_num}"
                    except (ValueError, TypeError):
                        page_info = f"page {metadata['page_num']}"
                elif 'page' in metadata and metadata['page'] is not None:
                    try:
                        page_num = int(float(metadata['page']))
                        page_info = f"page {page_num}"
                    except (ValueError, TypeError):
                        page_info = f"page {metadata['page']}"
            
            # Créer la source formatée
            if page_info:
                source_formatted = f"{doc_name}, {page_info}"
            else:
                source_formatted = doc_name
            
            priority_sources.append(source_formatted)
    
    # Supprimer les doublons tout en gardant l'ordre
    seen = set()
    unique_sources = []
    for source in priority_sources:
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)
    
    return unique_sources

# =============================================================================
# FONCTION POUR UTILISER LES SOURCES DE DOCUMENTS (PRIORITÉ 1)
# =============================================================================

def use_document_sources(response_content, priority_sources):
    """Utilise les sources de documents en supprimant les sources LLM existantes."""
    
    import re
    
    # Supprimer toutes les sections sources existantes de la réponse LLM
    # Pattern pour capturer les sections sources multiples
    patterns_to_remove = [
        r'📚\s*\*?\*?Sources?\s+utilisées?\s*:.*?(?=\n\n|\Z)',
        r'\*\*?📚.*?Sources?\s+utilisées?\s*:?\*?\*?.*?(?=\n\n|\Z)',
        r'Sources?\s+utilisées?\s*:.*?(?=\n\n|\Z)',
    ]
    
    cleaned_content = response_content
    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Nettoyer les espaces multiples et lignes vides
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.strip()
    
    # Ajouter la section sources de documents
    sources_section = "\n\n📚 **Sources utilisées :**\n"
    for source in priority_sources:
        sources_section += f"• {source}\n"
    
    final_content = cleaned_content + sources_section
    
    return final_content

# =============================================================================
# FONCTION POUR PRÉSERVER LES SOURCES LLM (PRIORITÉ 2)
# =============================================================================

def preserve_llm_sources(response_content):
    """Préserve les sources générées par le LLM quand pas de sources de documents."""
    
    import re
    
    # Vérifier si le LLM a généré des sources
    has_llm_sources = bool(re.search(r'📚|Sources?\s+utilisées?', response_content, re.IGNORECASE))
    
    if has_llm_sources:
        print("✅ Sources LLM détectées - conservation")
        
        # Nettoyer le format des sources LLM pour uniformiser
        # Remplacer les * par des • pour cohérence
        formatted_content = re.sub(r'(\n\s*)\*(\s+)', r'\1•\2', response_content)
        
        # S'assurer que la section sources a le bon format
        formatted_content = re.sub(
            r'📚\s*\*?\*?Sources?\s+utilisées?\s*:?',
            '📚 **Sources utilisées :**',
            formatted_content,
            flags=re.IGNORECASE
        )
        
        return formatted_content
    
    else:
        print("⚠️ Aucune source LLM détectée - ajout note explicative")
        
        # Ajouter une note explicative
        note_section = "\n\n📚 **Sources utilisées :**\n• Connaissances générales ANSD (aucun document spécifique fourni)\n"
        return response_content + note_section

# =============================================================================
# FONCTION DE DIAGNOSTIC DES SOURCES
# =============================================================================

def diagnose_sources(documents, response_content):
    """Diagnostique les sources disponibles pour débogage."""
    
    print("\n🔍 DIAGNOSTIC DES SOURCES:")
    print("="*50)
    
    # Analyser les documents
    print(f"📄 Documents fournis: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        print(f"   Document {i}:")
        print(f"      pdf_name: {metadata.get('pdf_name', 'Non défini')}")
        print(f"      page_num: {metadata.get('page_num', 'Non défini')}")
        print(f"      source: {metadata.get('source', 'Non défini')}")
    
    # Analyser les sources prioritaires
    priority_sources = extract_priority_sources(documents)
    print(f"\n🔝 Sources prioritaires extraites: {len(priority_sources)}")
    for i, source in enumerate(priority_sources, 1):
        print(f"   {i}. {source}")
    
    # Analyser les sources dans la réponse LLM
    import re
    llm_sources = re.findall(r'📚.*?Sources.*?:(.*?)(?=\n\n|\Z)', response_content, re.DOTALL | re.IGNORECASE)
    print(f"\n🤖 Sources LLM détectées: {len(llm_sources)}")
    for i, sources_block in enumerate(llm_sources, 1):
        print(f"   Bloc {i}: {sources_block.strip()[:100]}...")
    
    print("="*50)

# =============================================================================
# FONCTION GENERATE AVEC DIAGNOSTIC (VERSION DEBUG)
# =============================================================================

async def generate_with_debug(state: GraphState, *, config: RagConfiguration):
    """Version avec diagnostic pour déboguer les sources."""
    
    print("🤖 ---GENERATE AVEC DEBUG SOURCES---")
    
    messages = state.messages
    documents = state.documents
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVED_ANSD_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    context = format_docs_with_rich_metadata(documents)
    
    try:
        rag_chain = prompt | model
        response = await rag_chain.ainvoke({
            "context": context,
            "messages": messages
        })
        
        response_content = response.content
        
        # DIAGNOSTIC COMPLET
        diagnose_sources(documents, response_content)
        
        # Appliquer la logique de priorité
        priority_sources = extract_priority_sources(documents)
        
        if priority_sources:
            print("🔝 STRATÉGIE: Utilisation des sources de documents")
            final_response = use_document_sources(response_content, priority_sources)
        else:
            print("🤖 STRATÉGIE: Conservation des sources LLM")
            final_response = preserve_llm_sources(response_content)
        
        from langchain_core.messages import AIMessage
        enhanced_response = AIMessage(content=final_response)
        
        return {"messages": [enhanced_response], "documents": documents}
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="❌ Erreur technique ANSD.")
        return {"messages": [fallback], "documents": documents}


# =============================================================================
# CONFIGURATION DU WORKFLOW (NE PAS MODIFIER)
# =============================================================================

workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "ImprovedSimpleRag"