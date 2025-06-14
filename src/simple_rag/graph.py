# =============================================================================
# FICHIER 1: src/simple_rag/graph.py
# =============================================================================
# REMPLACEZ TOUT LE CONTENU DE CE FICHIER PAR LE CODE CI-DESSOUS

### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage
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

def format_docs_with_rich_metadata(docs) -> str:
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

async def generate(state: GraphState, *, config: RagConfiguration):
    """Génération de réponse améliorée avec sources explicites à la fin."""
    
    print("🤖 ---GENERATE AMÉLIORÉ AVEC SOURCES VISIBLES---")
    
    messages = state.messages
    documents = state.documents
    
    # Prompt spécialisé ANSD avec format tirets
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVED_ANSD_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    
    # Formatage enrichi des documents avec pages
    context = format_docs_with_rich_metadata(documents)
    
    # Créer un résumé des sources pour la réponse finale
    sources_for_response = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Extraire le nom du document
        doc_name = "Document ANSD"
        if 'pdf_name' in metadata and metadata['pdf_name']:
            doc_name = metadata['pdf_name']
        elif 'source' in metadata and metadata['source']:
            doc_name = metadata['source']
        
        # Extraire la page
        page_info = "page non spécifiée"
        if 'page_num' in metadata and metadata['page_num'] is not None:
            page_info = f"page {metadata['page_num']}"
        elif 'page' in metadata and metadata['page'] is not None:
            page_info = f"page {metadata['page']}"
        
        # Nettoyer le nom du document (enlever les chemins)
        if '/' in doc_name:
            doc_name = doc_name.split('/')[-1]
        if '\\' in doc_name:
            doc_name = doc_name.split('\\')[-1]
        
        sources_for_response.append(f"📄 {doc_name}, {page_info}")
    
    print(f"📄 Contexte généré ({len(context)} caractères)")
    print(f"📚 Sources identifiées: {len(documents)} documents")
    
    # Génération avec gestion d'erreurs
    try:
        rag_chain = prompt | model
        
        response = await rag_chain.ainvoke({
            "context": context,
            "messages": messages
        })
        
        # Ajouter les sources à la fin de la réponse
        response_content = response.content
        
        if sources_for_response:
            sources_section = "\n\n📚 **Sources utilisées :**\n"
            for source in sources_for_response:
                sources_section += f"• {source}\n"
            
            response_content += sources_section
        
        # Créer une nouvelle réponse avec les sources
        from langchain_core.messages import AIMessage
        enhanced_response = AIMessage(content=response_content)
        
        print(f"✅ Réponse générée ({len(response_content)} caractères)")
        print(f"👀 Aperçu: {response_content[:150]}...")
        
        return {"messages": [enhanced_response], "documents": documents}
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        
        # Réponse de fallback
        from langchain_core.messages import AIMessage
        fallback_response = AIMessage(content=
            "❌ Désolé, je rencontre des difficultés techniques pour analyser les documents ANSD. "
            "Veuillez reformuler votre question ou consulter directement www.ansd.sn pour les statistiques officielles."
        )
        
        return {"messages": [fallback_response], "documents": documents}
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