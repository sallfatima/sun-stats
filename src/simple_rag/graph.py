# =============================================================================
# FICHIER CORRIGÉ: src/simple_rag/graph.py
# =============================================================================

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
[Développez largement avec toutes les informations complémentaires pertinentes, sans limitation de longueur.]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation, changements méthodologiques, définitions spécifiques]

DOCUMENTS ANSD DISPONIBLES :
{context}

Analysez maintenant ces documents et répondez à la question de l'utilisateur de manière complète et approfondie."""

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

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
# FONCTIONS PRINCIPALES CORRIGÉES
# =============================================================================

async def retrieve(state, *, config):
    """Fonction de récupération corrigée pour gérer dict/dataclass"""
    print("🔍 ---RETRIEVE AVEC SUPPORT VISUEL---")
    
    # CORRECTION 1: Gestion hybride dict/dataclass
    if isinstance(state, dict):
        messages = state.get("messages", [])
        print("📝 State reçu comme dictionnaire")
    else:
        messages = getattr(state, "messages", [])
        print("📝 State reçu comme dataclass")
    
    # Extraction de la question
    question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            question = msg.content
            break
    
    if not question:
        print("❌ Aucune question trouvée")
        return {"documents": []}
    
    print(f"📝 Question: {question}")
    
    try:
        print("📄 Récupération documents textuels...")
        
        # CORRECTION 2: Configuration Pinecone sûre sans lambda_mult
        safe_config = dict(config) if config else {}
        if 'configurable' not in safe_config:
            safe_config['configurable'] = {}
        
        # Paramètres Pinecone compatibles
        safe_search_kwargs = {
            "k": 10,
           
            # Suppression de lambda_mult qui cause l'erreur
        }
        safe_config['configurable']['search_kwargs'] = safe_search_kwargs
        
        # Utilisation du retriever
        async with retrieval.make_retriever(safe_config) as retriever:
            documents = await retriever.ainvoke(question, safe_config)
            
            print(f"✅ Documents récupérés: {len(documents)}")
            
            # CORRECTION 3: Conversion en format approprié pour le state
            if documents and hasattr(documents[0], 'page_content'):
                # Garder les objets Document complets pour generate()
                return {"documents": documents}
            else:
                return {"documents": documents if documents else []}
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        import traceback
        traceback.print_exc()
        return {"documents": []}


async def generate(state, *, config):
    """Génération avec logique séquentielle CORRIGÉE."""
    
    print("🤖 ---GENERATE AVEC LOGIQUE SÉQUENTIELLE CORRIGÉE---")
    
    # CORRECTION 1: Gestion hybride dict/dataclass
    if isinstance(state, dict):
        messages = state.get("messages", [])
        documents = state.get("documents", [])
        print("📝 State reçu comme dictionnaire")
    else:
        messages = getattr(state, "messages", [])
        documents = getattr(state, "documents", [])
        print("📝 State reçu comme dataclass")
    
    # CORRECTION 2: Import des modules nécessaires
    try:
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return {"messages": [AIMessage(content="❌ Erreur de configuration ANSD.")], "documents": documents}
    
    # Extraire la question
    user_question = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            user_question = msg.content
            break
    
    print(f"❓ Question: {user_question}")
    print(f"📄 Documents disponibles: {len(documents)}")
    
    try:
        # =============================================================================
        # ÉTAPE 1 : ESSAYER AVEC LES DOCUMENTS INDEXÉS
        # =============================================================================
        
        print("\n🔍 ÉTAPE 1 : Recherche dans les documents indexés...")
        
        if documents:
            # Prompt pour utiliser les documents
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

DOCUMENTS DISPONIBLES :
{context}"""),
                ("placeholder", "{messages}")
            ])
            
            context = format_docs_with_metadata(documents)
            
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
        print(f"❌ ERREUR GÉNÉRATION: {e}")
        import traceback
        traceback.print_exc()
        
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
        'has_numbers': bool(re.search(r'\d+', response_content)),
        'has_content': len(response_content) > 100,
        'has_words': len(response_content.split()) > 20,
        'has_structure': '**' in response_content or '-' in response_content,
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
    
    # SEUIL RÉDUIT : 2 critères au lieu de 3
    is_satisfactory = success_count >= 2
    
    print(f"   📈 Score: {success_count}/6 critères")
    print(f"   🎯 Résultat: {'✅ SATISFAISANT' if is_satisfactory else '❌ INSUFFISANT'}")
    
    return is_satisfactory

def create_document_sources(documents, response_content):
    """Crée la section sources pour les documents indexés."""
    
    sources_section = "\n\n📚 **Sources utilisées (Documents indexés):**\n"
    
    # Extraire les documents réellement pertinents
    relevant_docs = []
    response_lower = response_content.lower()
    
    for doc in documents:
        if hasattr(doc, 'page_content'):
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
    
    sources_section = "\n\n📚 **Sources officielles :**\n"
    
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