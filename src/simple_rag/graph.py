# =============================================================================
# FICHIER: src/simple_rag/graph.py
# =============================================================================

"""
Système RAG simple amélioré pour l'ANSD avec support du contenu visuel.
"""

from typing import Dict, List, Any, Tuple
import re

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState

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
[Développez la réponse de manière complète et détaillée, sans limitation de longueur. Incluez tous les éléments pertinents pour une compréhension approfondie du sujet.]

**DONNÉES PRÉCISES :**
- Chiffres clés : [valeurs exactes avec unités]
- Années de référence : [périodes des données]
- Méthodologie : [type d'enquête, échantillon, méthode]
- Couverture géographique : [nationale, régionale, urbain/rural]

**CONTEXTE ADDITIONNEL :**
[Informations complémentaires, évolutions, comparaisons, explications méthodologiques]

**SOURCES :**
[Liste précise des documents et pages consultés]

Contexte: {context}"""

# =============================================================================
# FONCTIONS UTILITAIRES POUR LE SCORING ET LE PRÉTRAITEMENT
# =============================================================================

def preprocess_query(query: str) -> str:
    """
    Prétraite la requête utilisateur pour améliorer la recherche.
    
    Args:
        query: Requête originale de l'utilisateur
        
    Returns:
        Requête enrichie avec synonymes et termes ANSD
    """
    # Synonymes spécifiques ANSD
    ansd_synonyms = {
        'population': ['habitants', 'démographie', 'peuplement', 'résidents'],
        'ménages': ['foyers', 'familles', 'unités résidentielles'],
        'pauvreté': ['indigence', 'précarité', 'conditions de vie'],
        'emploi': ['travail', 'activité économique', 'occupation', 'profession'],
        'éducation': ['scolarisation', 'alphabétisation', 'instruction'],
        'santé': ['morbidité', 'mortalité', 'état sanitaire'],
        'urbain': ['ville', 'citadin', 'agglomération'],
        'rural': ['campagne', 'agricole', 'villageois'],
        'région': ['administrative', 'territoire', 'zone géographique'],
        'taux': ['pourcentage', 'proportion', 'ratio'],
        'évolution': ['tendance', 'progression', 'changement'],
        'répartition': ['distribution', 'ventilation', 'structure']
    }
    
    # Enrichir la requête
    enriched_terms = [query.lower()]
    
    for keyword, synonyms in ansd_synonyms.items():
        if keyword in query.lower():
            enriched_terms.extend(synonyms[:2])  # Ajouter 2 synonymes max
    
    # Ajouter des termes contextuels ANSD
    if any(term in query.lower() for term in ['population', 'habitants', 'recensement']):
        enriched_terms.append('rgph')
    
    if any(term in query.lower() for term in ['pauvreté', 'conditions', 'vie']):
        enriched_terms.append('esps ehcvm')
    
    if any(term in query.lower() for term in ['emploi', 'chômage', 'activité']):
        enriched_terms.append('enes')
    
    if any(term in query.lower() for term in ['santé', 'démographique']):
        enriched_terms.append('eds')
    
    return ' '.join(enriched_terms)


def score_documents_relevance(documents: List[Any], query: str) -> List[Tuple[float, Any]]:
    """
    Score les documents selon leur pertinence pour la requête.
    
    Args:
        documents: Liste des documents à scorer
        query: Requête originale
        
    Returns:
        Liste de tuples (score, document) triée par score décroissant
    """
    scored_documents = []
    query_lower = query.lower()
    
    # Mots-clés importants pour l'ANSD
    important_keywords = [
        'population', 'ménages', 'pauvreté', 'emploi', 'éducation', 
        'santé', 'région', 'urbain', 'rural', 'taux', 'pourcentage',
        'rgph', 'eds', 'esps', 'ehcvm', 'enes', 'ansd'
    ]
    
    for doc in documents:
        score = 0
        content = getattr(doc, 'page_content', '').lower()
        metadata = getattr(doc, 'metadata', {})
        
        # Score basé sur les mots-clés dans le contenu
        for keyword in important_keywords:
            if keyword in query_lower and keyword in content:
                score += 5
        
        # Score basé sur les métadonnées
        if 'survey_type' in metadata:
            survey_type = metadata['survey_type'].lower()
            if survey_type in query_lower:
                score += 10
        
        # Score basé sur le type de document
        doc_type = metadata.get('type', '')
        if doc_type in ['visual_chart', 'visual_table']:
            score += 8  # Bonus pour le contenu visuel
        
        # Score basé sur la correspondance textuelle générale
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:
                score += content.count(word) * 2
        
        scored_documents.append((score, doc))
    
    # Trier par score décroissant
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    return scored_documents


def format_docs_with_rich_metadata(docs: List[Any]) -> str:
    """
    Formate les documents avec métadonnées enrichies pour le prompt.
    
    Args:
        docs: Documents à formater
        
    Returns:
        Contexte formaté avec métadonnées riches
    """
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
        if 'page' in metadata:
            source_info.append(f"📖 Page: {metadata['page']}")
        if 'type' in metadata:
            source_info.append(f"🔖 Type: {metadata['type']}")
        if 'caption' in metadata:
            source_info.append(f"📊 Titre: {metadata['caption']}")
        
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
        if metadata.get('type') in ['visual_chart', 'visual_table']:
            content_indicators.append("🎨 VISUEL")
        
        if content_indicators:
            header += f"🏷️ Catégories: {' | '.join(content_indicators)}\n{'-'*50}\n"
        
        formatted_parts.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted_parts)


def analyze_retrieved_documents(documents: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Analyse et sépare les documents textuels des documents visuels.
    
    Args:
        documents: Liste des documents récupérés
        
    Returns:
        Tuple (documents_textuels, documents_visuels)
    """
    text_docs = []
    visual_docs = []
    
    for doc in documents:
        metadata = getattr(doc, 'metadata', {})
        doc_type = metadata.get('type', '')
        source_type = metadata.get('source_type', '')
        
        # Identifier les documents visuels
        if (doc_type in ['visual_chart', 'visual_table'] or 
            source_type == 'visual' or
            'image_path' in metadata or 
            'table_path' in metadata):
            visual_docs.append(doc)
        else:
            text_docs.append(doc)
    
    return text_docs, visual_docs


def enrich_query_for_visual_content(question: str) -> str:
    """
    Enrichit la requête pour améliorer la récupération de contenu visuel.
    
    Args:
        question: Question originale de l'utilisateur
        
    Returns:
        Requête enrichie pour la recherche
    """
    # Mots-clés qui suggèrent du contenu visuel
    visual_keywords = {
        'répartition': ['graphique', 'diagramme', 'pourcentage'],
        'évolution': ['courbe', 'tendance', 'graphique'],
        'comparaison': ['tableau', 'données', 'statistiques'],
        'taux': ['pourcentage', 'graphique', 'données'],
        'structure': ['répartition', 'graphique', 'diagramme'],
        'données': ['tableau', 'statistiques', 'chiffres'],
        'statistiques': ['tableau', 'données', 'chiffres'],
        'pourcentage': ['répartition', 'graphique', 'diagramme']
    }
    
    # Construire la requête enrichie
    query_parts = [question]
    
    question_lower = question.lower()
    
    # Ajouter des termes de recherche visuelle pertinents
    for keyword, related_terms in visual_keywords.items():
        if keyword in question_lower:
            query_parts.extend(related_terms[:2])  # Ajouter 2 termes liés max
    
    # Ajouter des termes génériques pour le contenu visuel
    if any(term in question_lower for term in ['répartition', 'évolution', 'taux', 'pourcentage']):
        query_parts.append('visual_chart')
        query_parts.append('graphique')
    
    if any(term in question_lower for term in ['données', 'tableau', 'statistiques']):
        query_parts.append('visual_table')
        query_parts.append('tableau')
    
    return ' '.join(query_parts)


# =============================================================================
# FONCTIONS PRINCIPALES DU WORKFLOW
# =============================================================================

async def retrieve(state: GraphState, *, config: RagConfiguration):
    """Récupération améliorée avec support du contenu visuel et scoring intelligent."""
    
    print("🔍 ---RETRIEVE AVEC SUPPORT VISUEL AMÉLIORÉ---")
    
    # Extraire la question
    last_message = state.messages[-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    print(f"❓ Question originale: {question}")
    
    # Enrichir la requête pour le contenu visuel
    enriched_query = enrich_query_for_visual_content(question)
    
    # Prétraitement additionnel
    processed_query = preprocess_query(enriched_query)
    
    print(f"🔍 Requête enrichie: {processed_query}")
    
    # Configuration de recherche optimisée
    with retrieval.make_retriever(config) as retriever:
        # Recherche avec la requête enrichie
        documents = retriever.invoke(processed_query)
    
    print(f"📄 Documents récupérés: {len(documents)}")
    
    # Analyser et classer les documents
    text_docs, visual_docs = analyze_retrieved_documents(documents)
    
    print(f"📝 Documents textuels: {len(text_docs)}")
    print(f"🎨 Documents visuels: {len(visual_docs)}")
    
    # Afficher les détails des documents visuels trouvés
    if visual_docs:
        print("🎨 Éléments visuels détectés:")
        for i, doc in enumerate(visual_docs, 1):
            metadata = getattr(doc, 'metadata', {})
            doc_type = metadata.get('type', 'unknown')
            caption = metadata.get('caption', 'Sans titre')
            print(f"   {i}. {doc_type}: {caption}")
    
    # Combiner tous les documents pour le processing suivant
    all_documents = text_docs + visual_docs
    
    # Appliquer un scoring intelligent pour prioriser les documents pertinents
    scored_documents = score_documents_relevance(all_documents, question)
    
    # Prendre les 20 meilleurs documents
    best_documents = [doc for score, doc in scored_documents[:20]]
    
    print(f"✅ Documents sélectionnés après scoring: {len(best_documents)}")
    
    return {"documents": best_documents, "messages": state.messages}


async def generate(state: GraphState, *, config: RagConfiguration):
    """Génération avec support d'affichage automatique du contenu visuel."""
    
    print("🤖 ---GENERATE AVEC SUPPORT VISUEL---")
    
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
    print(f"📄 Total documents: {len(documents)}")
    
    # Séparer le contenu textuel et visuel
    text_docs, visual_docs = analyze_retrieved_documents(documents)
    
    print(f"📝 Documents textuels pour génération: {len(text_docs)}")
    print(f"🎨 Documents visuels détectés: {len(visual_docs)}")
    
    # Générer la réponse basée sur le contenu textuel et visuel combiné
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVED_ANSD_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # Utiliser tous les documents (textuels + visuels) pour le contexte
    context = format_docs_with_rich_metadata(documents)
    
    response = await (prompt | model).ainvoke({
        "context": context,
        "question": user_question
    })
    
    textual_response = response.content
    
    # Enrichir la réponse si du contenu visuel est disponible
    if visual_docs:
        # Ajouter une mention des éléments visuels disponibles
        visual_summary = create_visual_content_summary(visual_docs)
        
        enhanced_response = f"""{textual_response}

📊 **Éléments visuels disponibles :**
{visual_summary}

*Les graphiques et tableaux correspondants seront affichés automatiquement dans le chat.*"""
        
        final_response = enhanced_response
        
        # Marquer la réponse avec des métadonnées pour l'affichage visuel
        response_metadata = {
            'has_visual_content': True,
            'visual_count': len(visual_docs),
            'visual_types': [doc.metadata.get('type', 'unknown') for doc in visual_docs]
        }
        
    else:
        final_response = textual_response
        response_metadata = {'has_visual_content': False}
    
    # Créer la réponse finale
    enhanced_response = AIMessage(
        content=final_response,
        response_metadata=response_metadata
    )
    
    print(f"✅ Réponse générée avec {len(visual_docs)} éléments visuels")
    
    return {"messages": [enhanced_response], "documents": documents}


def create_visual_content_summary(visual_docs: List[Any]) -> str:
    """
    Crée un résumé textuel des éléments visuels disponibles.
    
    Args:
        visual_docs: Documents visuels
        
    Returns:
        Résumé formaté des éléments visuels
    """
    summary_parts = []
    
    charts_count = 0
    tables_count = 0
    
    for i, doc in enumerate(visual_docs, 1):
        metadata = getattr(doc, 'metadata', {})
        doc_type = metadata.get('type', 'unknown')
        caption = metadata.get('caption', f'Élément {i}')
        pdf_name = metadata.get('pdf_name', 'Document ANSD')
        page = metadata.get('page', '')
        
        if doc_type == 'visual_chart':
            charts_count += 1
            icon = "📊"
        elif doc_type == 'visual_table':
            tables_count += 1
            icon = "📋"
        else:
            icon = "📄"
        
        source_info = f"{pdf_name}"
        if page:
            source_info += f" (page {page})"
        
        summary_parts.append(f"{icon} **{caption}** - *{source_info}*")
    
    # Ajouter un résumé en en-tête
    summary_header = []
    if charts_count > 0:
        summary_header.append(f"{charts_count} graphique(s)")
    if tables_count > 0:
        summary_header.append(f"{tables_count} tableau(x)")
    
    if summary_header:
        header = " et ".join(summary_header)
        summary_parts.insert(0, f"*{header} trouvé(s) :*\n")
    
    return "\n".join(summary_parts)


# =============================================================================
# FONCTIONS DE DEBUG ET DIAGNOSTIC (CONSERVÉES DE VOTRE CODE)
# =============================================================================

def diagnose_sources(documents: List[Any], response_content: str) -> None:
    """Diagnostic des sources pour déboguer."""
    
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
        print(f"      type: {metadata.get('type', 'Non défini')}")
    
    print("="*50)


def extract_priority_sources(documents: List[Any]) -> List[str]:
    """Extrait les sources prioritaires des documents."""
    
    priority_sources = []
    
    for doc in documents:
        metadata = getattr(doc, 'metadata', {})
        
        # Construire la source à partir des métadonnées
        source_parts = []
        
        if 'pdf_name' in metadata:
            source_parts.append(metadata['pdf_name'])
        
        if 'page' in metadata:
            source_parts.append(f"page {metadata['page']}")
        elif 'page_num' in metadata:
            source_parts.append(f"page {metadata['page_num']}")
        
        if source_parts:
            priority_sources.append(", ".join(source_parts))
    
    return list(set(priority_sources))  # Supprimer les doublons


def use_document_sources(response_content: str, priority_sources: List[str]) -> str:
    """Utilise les sources des documents dans la réponse."""
    
    if priority_sources:
        sources_section = f"\n\n📚 **Sources consultées :**\n"
        for i, source in enumerate(priority_sources[:5], 1):  # Limiter à 5 sources
            sources_section += f"• {source}\n"
        
        return response_content + sources_section
    
    return response_content


def preserve_llm_sources(response_content: str) -> str:
    """Préserve les sources générées par le LLM."""
    return response_content


# =============================================================================
# VERSION DEBUG AVEC DIAGNOSTIC COMPLET (CONSERVÉE)
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
        
        enhanced_response = AIMessage(content=final_response)
        
        return {"messages": [enhanced_response], "documents": documents}
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        fallback = AIMessage(content="❌ Erreur technique ANSD.")
        return {"messages": [fallback], "documents": documents}


# =============================================================================
# CONFIGURATION DU WORKFLOW
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