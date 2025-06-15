# src/simple_rag/graph_enhanced.py
"""
Simple RAG Graph amélioré avec support complet pour l'affichage de graphiques et tableaux
"""

from __future__ import annotations
import chainlit as cl
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import base64
import json
import re
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from shared.configuration import RagConfiguration
from shared.llm import load_chat_model
from shared.vectorstore import load_vectorstore


# =============================================================================
# ÉTAT DU GRAPHE AMÉLIORÉ
# =============================================================================

class GraphState(TypedDict):
    """État amélioré du graphe avec support visuel."""
    messages: Annotated[list, add_messages]
    documents: List[Any]
    visual_elements: List[Dict[str, Any]]  # Nouveau : éléments visuels
    has_visual_content: bool               # Nouveau : indicateur de contenu visuel
    response_metadata: Dict[str, Any]      # Nouveau : métadonnées de réponse


class InputState(TypedDict):
    """État d'entrée du graphe."""
    messages: Annotated[list, add_messages]


# =============================================================================
# FONCTIONS D'AFFICHAGE VISUEL AMÉLIORÉES
# =============================================================================

async def display_chart_from_metadata(metadata: Dict[str, Any], caption: str = "") -> bool:
    """
    Affiche un graphique basé sur ses métadonnées.
    
    Args:
        metadata: Métadonnées du document contenant l'image_path
        caption: Légende du graphique
        
    Returns:
        True si affichage réussi, False sinon
    """
    try:
        image_path = metadata.get('image_path')
        if not image_path:
            return False
            
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"⚠️ Image non trouvée: {image_path}")
            return False
        
        # Préparer les informations du graphique
        pdf_name = metadata.get('pdf_name', metadata.get('source', 'Document ANSD'))
        page = metadata.get('page', metadata.get('page_num', 'N/A'))
        chart_type = metadata.get('chart_type', 'graphique')
        
        # Titre enrichi
        if not caption:
            caption = metadata.get('caption', f"Graphique de {pdf_name}")
        
        title = f"📊 **{chart_type.title()}** : {caption}"
        
        # Informations de source
        source_info = f"*Source : {pdf_name}"
        if page != 'N/A':
            source_info += f", page {page}"
        source_info += "*"
        
        # Message avec titre et source
        await cl.Message(
            content=f"{title}\n\n{source_info}"
        ).send()
        
        # Afficher l'image
        elements = [cl.Image(name="chart", path=str(image_file), display="inline")]
        await cl.Message(content="", elements=elements).send()
        
        # Afficher les indicateurs numériques si disponibles
        indicators = metadata.get('numerical_indicators', [])
        if indicators:
            indicators_text = "📈 **Indicateurs clés :**\n"
            for ind in indicators[:5]:  # Limiter à 5 indicateurs
                indicators_text += f"• {ind.get('type', 'Valeur')} : {ind.get('value', 'N/A')}\n"
            await cl.Message(content=indicators_text).send()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur affichage graphique: {e}")
        return False


async def display_table_from_metadata(metadata: Dict[str, Any], content: str = "") -> bool:
    """
    Affiche un tableau basé sur ses métadonnées et contenu.
    
    Args:
        metadata: Métadonnées du document
        content: Contenu textuel du tableau
        
    Returns:
        True si affichage réussi, False sinon
    """
    try:
        # Informations du tableau
        pdf_name = metadata.get('pdf_name', metadata.get('source', 'Document ANSD'))
        page = metadata.get('page', metadata.get('page_num', 'N/A'))
        caption = metadata.get('caption', 'Tableau ANSD')
        
        # Titre du tableau
        title = f"📋 **Tableau** : {caption}"
        
        # Informations de source
        source_info = f"*Source : {pdf_name}"
        if page != 'N/A':
            source_info += f", page {page}"
        source_info += "*"
        
        # Message avec titre et source
        await cl.Message(
            content=f"{title}\n\n{source_info}"
        ).send()
        
        # Formatter le contenu du tableau
        if content:
            formatted_table = format_table_content(content, metadata)
            await cl.Message(content=formatted_table).send()
        
        # Afficher les statistiques si disponibles
        stats = metadata.get('table_stats', {})
        if stats:
            stats_text = "📊 **Statistiques du tableau :**\n"
            for key, value in stats.items():
                stats_text += f"• {key.replace('_', ' ').title()} : {value}\n"
            await cl.Message(content=stats_text).send()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur affichage tableau: {e}")
        return False


def format_table_content(content: str, metadata: Dict[str, Any]) -> str:
    """
    Formate le contenu d'un tableau pour un affichage optimal.
    
    Args:
        content: Contenu brut du tableau
        metadata: Métadonnées pour le contexte
        
    Returns:
        Contenu formaté en markdown
    """
    if not content:
        return "```\nContenu du tableau non disponible\n```"
    
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.strip():
            # Nettoyer et formatter les lignes
            if '|' in line:
                # Déjà formaté en style tableau
                formatted_lines.append(line)
            elif line.startswith('Colonnes:'):
                # En-têtes de colonnes
                headers = line.replace('Colonnes:', '').strip()
                formatted_lines.append(f"| {headers.replace(' | ', ' | ')} |")
                # Ajouter la ligne de séparation markdown
                header_count = headers.count('|') + 1
                formatted_lines.append('|' + ' --- |' * header_count)
            elif line.startswith('Ligne'):
                # Données du tableau
                data = re.sub(r'Ligne \d+:', '', line).strip()
                formatted_lines.append(f"| {data.replace(' | ', ' | ')} |")
            else:
                formatted_lines.append(line)
    
    if formatted_lines:
        return f"```markdown\n" + '\n'.join(formatted_lines) + "\n```"
    else:
        return f"```\n{content}\n```"


# =============================================================================
# FONCTIONS D'ANALYSE ET DÉTECTION VISUELLES
# =============================================================================

def extract_visual_elements(documents: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Sépare les documents textuels des éléments visuels.
    
    Args:
        documents: Liste des documents récupérés
        
    Returns:
        Tuple (documents_textuels, éléments_visuels)
    """
    text_docs = []
    visual_elements = []
    
    for doc in documents:
        if not hasattr(doc, 'metadata') or not doc.metadata:
            text_docs.append(doc)
            continue
            
        doc_type = doc.metadata.get('type', '')
        
        if doc_type in ['visual_chart', 'visual_table']:
            visual_element = {
                'type': doc_type,
                'content': getattr(doc, 'page_content', str(doc)),
                'metadata': doc.metadata,
                'document': doc
            }
            visual_elements.append(visual_element)
        else:
            text_docs.append(doc)
    
    return text_docs, visual_elements


async def process_and_display_visual_elements(visual_elements: List[Dict[str, Any]], 
                                            user_question: str) -> bool:
    """
    Traite et affiche tous les éléments visuels pertinents.
    
    Args:
        visual_elements: Liste des éléments visuels
        user_question: Question de l'utilisateur pour le contexte
        
    Returns:
        True si des éléments ont été affichés
    """
    if not visual_elements:
        return False
    
    print(f"🎨 Traitement de {len(visual_elements)} éléments visuels...")
    
    # Message d'introduction
    intro_msg = f"📊 **Contenu visuel ANSD pertinent**\n*En rapport avec : {user_question}*\n"
    await cl.Message(content=intro_msg).send()
    
    charts_displayed = 0
    tables_displayed = 0
    
    for i, element in enumerate(visual_elements, 1):
        element_type = element['type']
        metadata = element['metadata']
        content = element['content']
        
        try:
            if element_type == 'visual_chart':
                success = await display_chart_from_metadata(metadata)
                if success:
                    charts_displayed += 1
                    
            elif element_type == 'visual_table':
                success = await display_table_from_metadata(metadata, content)
                if success:
                    tables_displayed += 1
                    
        except Exception as e:
            print(f"❌ Erreur affichage élément {i}: {e}")
            continue
    
    # Message de résumé
    if charts_displayed > 0 or tables_displayed > 0:
        summary = f"✅ **Affichage terminé** : {charts_displayed} graphiques, {tables_displayed} tableaux"
        await cl.Message(content=summary).send()
        return True
    
    return False


def analyze_visual_relevance(visual_elements: List[Dict[str, Any]], 
                           user_question: str) -> List[Dict[str, Any]]:
    """
    Analyse la pertinence des éléments visuels par rapport à la question.
    
    Args:
        visual_elements: Liste des éléments visuels
        user_question: Question de l'utilisateur
        
    Returns:
        Liste filtrée et classée par pertinence
    """
    if not visual_elements:
        return []
    
    # Mots-clés de la question
    question_keywords = set(re.findall(r'\b\w+\b', user_question.lower()))
    
    scored_elements = []
    
    for element in visual_elements:
        score = 0
        metadata = element['metadata']
        content = element['content']
        
        # Score basé sur la caption
        caption = metadata.get('caption', '').lower()
        if caption:
            caption_keywords = set(re.findall(r'\b\w+\b', caption))
            score += len(question_keywords.intersection(caption_keywords)) * 3
        
        # Score basé sur le contenu
        if content:
            content_keywords = set(re.findall(r'\b\w+\b', content.lower()))
            score += len(question_keywords.intersection(content_keywords)) * 2
        
        # Score basé sur le type de document
        pdf_name = metadata.get('pdf_name', '').lower()
        if pdf_name:
            pdf_keywords = set(re.findall(r'\b\w+\b', pdf_name))
            score += len(question_keywords.intersection(pdf_keywords)) * 1
        
        # Bonus pour certains types de questions
        if any(word in user_question.lower() for word in ['graphique', 'tableau', 'figure', 'chart']):
            score += 5
        
        if score > 0:
            element['relevance_score'] = score
            scored_elements.append(element)
    
    # Trier par score décroissant
    scored_elements.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Limiter à 5 éléments les plus pertinents
    return scored_elements[:5]


# =============================================================================
# FONCTIONS PRINCIPALES DU GRAPHE
# =============================================================================

async def retrieve_enhanced(state: GraphState, *, config: RagConfiguration) -> Dict[str, Any]:
    """
    Fonction de récupération améliorée avec séparation du contenu visuel.
    """
    print("---RETRIEVE ENHANCED---")
    
    messages = state["messages"]
    last_message = messages[-1]
    user_question = last_message.content
    
    print(f"Question utilisateur: {user_question}")
    
    # Charger le vectorstore
    vectorstore = load_vectorstore(config)
    
    # Récupération des documents (augmentée pour capturer plus d'éléments visuels)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": min(15, getattr(config, 'max_docs', 10))}  # Augmenté pour les visuels
    )
    
    documents = await retriever.ainvoke(user_question)
    print(f"Documents récupérés: {len(documents)}")
    
    # Séparer les documents textuels et visuels
    text_docs, visual_elements = extract_visual_elements(documents)
    
    print(f"Documents textuels: {len(text_docs)}")
    print(f"Éléments visuels: {len(visual_elements)}")
    
    # Analyser la pertinence des éléments visuels
    relevant_visual_elements = analyze_visual_relevance(visual_elements, user_question)
    
    print(f"Éléments visuels pertinents: {len(relevant_visual_elements)}")
    
    return {
        "documents": text_docs,
        "visual_elements": relevant_visual_elements,
        "has_visual_content": len(relevant_visual_elements) > 0,
        "response_metadata": {
            "total_docs": len(documents),
            "text_docs": len(text_docs),
            "visual_elements": len(relevant_visual_elements),
            "query": user_question
        }
    }


async def generate_enhanced(state: GraphState, *, config: RagConfiguration) -> Dict[str, Any]:
    """
    Fonction de génération améliorée avec affichage automatique du contenu visuel.
    """
    print("---GENERATE ENHANCED---")
    
    messages = state["messages"]
    documents = state["documents"]
    visual_elements = state.get("visual_elements", [])
    has_visual = state.get("has_visual_content", False)
    
    # 1. Afficher d'abord le contenu visuel si disponible
    if has_visual and visual_elements:
        user_question = messages[-1].content
        await process_and_display_visual_elements(visual_elements, user_question)
    
    # 2. Générer la réponse textuelle
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un expert statisticien de l'ANSD du Sénégal.

Analysez les documents fournis et répondez à la question en utilisant EXCLUSIVEMENT ces documents.

RÈGLES :
✅ Utilisez TOUTES les informations pertinentes trouvées dans les documents
✅ Donnez les chiffres EXACTS trouvés
✅ Citez les sources précises (nom du document, page)
✅ Développez votre réponse avec les détails disponibles

FORMAT DE RÉPONSE :

**RÉPONSE DIRECTE :**
[Réponse détaillée basée sur les documents trouvés]

**DONNÉES PRÉCISES :**
- Chiffres exacts : [valeurs des documents]
- Année de référence : [année des documents]
- Méthodologie : [enquête/recensement des documents]

**CONTEXTE ADDITIONNEL :**
[Toutes les informations complémentaires trouvées dans les documents]

Si des graphiques ou tableaux ont été affichés ci-dessus, référencez-les dans votre réponse.

DOCUMENTS DISPONIBLES :
{context}"""),
        ("placeholder", "{messages}")
    ])
    
    # Formatage des documents avec métadonnées enrichies
    def format_docs_with_enhanced_metadata(docs):
        if not docs:
            return "Aucun document textuel pertinent trouvé."
        
        formatted_parts = []
        
        for i, doc in enumerate(docs, 1):
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
            
            formatted_parts.append(f"{header}\n{content}\n")
        
        return "\n".join(formatted_parts)
    
    # Chargement du modèle
    model = load_chat_model(config)
    
    # Génération de la réponse
    context = format_docs_with_enhanced_metadata(documents)
    
    rag_chain = prompt | model
    
    response = await rag_chain.ainvoke({
        "context": context,
        "messages": messages
    })
    
    # Enrichir la réponse avec des informations sur le contenu visuel
    if has_visual and visual_elements:
        visual_summary = f"\n\n📊 **Contenu visuel affiché :** {len(visual_elements)} éléments (graphiques et tableaux) ont été présentés ci-dessus en complément de cette analyse."
        response.content += visual_summary
    
    print(f"Réponse générée avec contenu visuel: {has_visual}")
    
    return {
        "messages": [response],
        "response_metadata": {
            **state.get("response_metadata", {}),
            "visual_displayed": has_visual,
            "visual_count": len(visual_elements)
        }
    }


# =============================================================================
# CONSTRUCTION DU GRAPHE
# =============================================================================

# Configuration du workflow
workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

# Définir les nœuds
workflow.add_node("retrieve", retrieve_enhanced)
workflow.add_node("generate", generate_enhanced)

# Construire le graphe
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compiler
graph = workflow.compile()
graph.name = "EnhancedSimpleRagWithVisuals"

# =============================================================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# =============================================================================

def get_visual_stats(state: GraphState) -> Dict[str, Any]:
    """Retourne les statistiques sur le contenu visuel traité."""
    metadata = state.get("response_metadata", {})
    return {
        "visual_elements_found": metadata.get("visual_elements", 0),
        "visual_displayed": metadata.get("visual_displayed", False),
        "total_documents": metadata.get("total_docs", 0),
        "text_documents": metadata.get("text_docs", 0)
    }


async def debug_visual_content(documents: List[Any]) -> None:
    """Fonction de debug pour analyser le contenu visuel disponible."""
    print("\n🔍 DEBUG VISUAL CONTENT")
    print("=" * 50)
    
    text_docs, visual_elements = extract_visual_elements(documents)
    
    print(f"📝 Documents textuels: {len(text_docs)}")
    print(f"🎨 Éléments visuels: {len(visual_elements)}")
    
    for i, element in enumerate(visual_elements, 1):
        metadata = element['metadata']
        print(f"\n📊 Élément visuel #{i}:")
        print(f"   Type: {element['type']}")
        print(f"   Caption: {metadata.get('caption', 'N/A')}")
        print(f"   Image path: {metadata.get('image_path', 'N/A')}")
        print(f"   PDF: {metadata.get('pdf_name', 'N/A')}")
        print(f"   Page: {metadata.get('page', 'N/A')}")

# Export du graphe
__all__ = ["graph", "get_visual_stats", "debug_visual_content"]