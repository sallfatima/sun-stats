import chainlit as cl
import sys
import os
from pathlib import Path
import asyncio
from typing import List, Dict, Any, Tuple

# Ajouter le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# =============================================================================
# IMPORTS SÉCURISÉS AVEC GESTION D'ERREURS
# =============================================================================

def safe_import_rag_graph():
    """Import sécurisé de RAGGraph avec gestion des erreurs LangGraph"""
    try:
        print("🔄 Tentative d'import simple_rag.graph...")
        from simple_rag.graph import graph as simple_rag_graph
        print("✅ Import LangGraph réussi")
        return simple_rag_graph, "langgraph"
    except Exception as e:
        print(f"⚠️ Erreur import LangGraph: {e}")
        try:
            print("🔄 Tentative d'import RAGGraph classe...")
            from simple_rag.graph import RAGGraph
            print("✅ Import classe RAGGraph réussi")
            return RAGGraph(), "class"
        except Exception as e2:
            print(f"❌ Erreur import classe RAGGraph: {e2}")
            print("🔄 Utilisation de DirectRAG comme fallback...")
            # Créer DirectRAG comme solution de secours
            return create_direct_rag_fallback(), "class"

def create_direct_rag_fallback():
    """Crée une instance DirectRAG comme solution de secours"""
    
    class DirectRAGFallback:
        def __init__(self):
            self.retriever = None
            self.llm = None
            self.setup()
        
        def setup(self):
            try:
                print("🔧 Configuration DirectRAG Fallback...")
                
                # Configuration LLM
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1
                )
                print("✅ LLM configuré")
                
                # Configuration Pinecone
                self.setup_pinecone_retriever()
                
            except Exception as e:
                print(f"❌ Erreur setup DirectRAG Fallback: {e}")
        
        def setup_pinecone_retriever(self):
            try:
                print("🔌 Configuration Pinecone...")
                from langchain_pinecone import PineconeVectorStore
                from langchain_openai import OpenAIEmbeddings
                
                embeddings = OpenAIEmbeddings()
                vectorstore = PineconeVectorStore.from_existing_index(
                    index_name=os.getenv('PINECONE_INDEX', 'index-ansd'),
                    embedding=embeddings
                )
                self.retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
                print("✅ Pinecone configuré")
                
            except Exception as e:
                print(f"❌ Erreur configuration Pinecone: {e}")
                self.retriever = None
        
        def is_ready(self):
            return self.llm is not None and self.retriever is not None
        
        async def ask(self, question: str, chat_history: list) -> tuple:
            try:
                if not self.is_ready():
                    return "❌ Système non configuré", []
                
                # Récupération documents
                print("🔍 Récupération documents Pinecone...")
                loop = asyncio.get_event_loop()
                
                def get_docs():
                    return self.retriever.get_relevant_documents(question)
                
                documents = await loop.run_in_executor(None, get_docs)
                print(f"📄 {len(documents)} documents récupérés")
                
                # Contexte
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content[:500]}"
                    for i, doc in enumerate(documents[:8])
                ]) if documents else "Aucun document trouvé."
                
                # Historique
                history_text = "\n\n".join([
                    f"Q: {q[:150]}\nR: {r[:200]}"
                    for q, r in chat_history[-2:]
                ]) if chat_history else ""
                
                # Prompt
                prompt = f"""Tu es un expert statisticien de l'ANSD du Sénégal.

DOCUMENTS ANSD :
{context}

HISTORIQUE :
{history_text}

QUESTION : {question}

Réponds uniquement basé sur les documents ANSD fournis. Cite tes sources précisément.

RÉPONSE :"""
                
                # Génération
                from langchain_core.messages import HumanMessage
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                
                return response.content, documents
                
            except Exception as e:
                return f"❌ Erreur DirectRAG: {str(e)}", []
    
    return DirectRAGFallback()

def safe_import_retrieval_graph():
    """Import sécurisé du retrieval graph"""
    try:
        from retrieval_graph.graph import graph as retrieval_graph
        return retrieval_graph, "langgraph"
    except ImportError as e:
        print(f"❌ Retrieval graph non disponible: {e}")
        return None, None

def safe_import_self_rag():
    """Import sécurisé du self RAG graph"""
    try:
        from self_rag.graph import graph as self_rag_graph
        return self_rag_graph, "langgraph"
    except ImportError as e:
        print(f"❌ Self RAG non disponible: {e}")
        return None, None

# Configuration des graphiques disponibles
AVAILABLE_GRAPHS = {}

# Initialisation des graphiques
simple_rag, simple_type = safe_import_rag_graph()
if simple_rag:
    AVAILABLE_GRAPHS["simple_rag"] = {
        "name": "Simple RAG",
        "description": "RAG avec Pinecone + support visuel",
        "instance": simple_rag,
        "type": simple_type
    }

retrieval_graph, retrieval_type = safe_import_retrieval_graph()
if retrieval_graph:
    AVAILABLE_GRAPHS["retrieval_graph"] = {
        "name": "Retrieval Graph", 
        "description": "RAG amélioré avec meilleure récupération",
        "instance": retrieval_graph,
        "type": retrieval_type
    }

self_rag, self_type = safe_import_self_rag()
if self_rag:
    AVAILABLE_GRAPHS["self_rag"] = {
        "name": "Self RAG",
        "description": "RAG avec auto-évaluation et correction",
        "instance": self_rag,
        "type": self_type
    }

print(f"📊 Graphiques disponibles: {list(AVAILABLE_GRAPHS.keys())}")

# =============================================================================
# FONCTIONS DE SUPPORT VISUEL
# =============================================================================

async def detect_and_display_visual_content(documents: List[Any], user_question: str) -> Tuple[List[Any], bool]:
    """
    Détecte et affiche automatiquement les éléments visuels pertinents
    """
    if not documents:
        return documents, False
    
    print(f"🔍 Analyse de {len(documents)} documents pour contenu visuel...")
    
    # Séparer documents textuels et visuels
    text_docs = []
    visual_elements = []
    
    for doc in documents:
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_type = doc.metadata.get('type', '')
            
            if doc_type in ['visual_chart', 'visual_table']:
                visual_elements.append({
                    'type': doc_type,
                    'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    'metadata': doc.metadata
                })
            else:
                text_docs.append(doc)
        else:
            text_docs.append(doc)
    
    print(f"📝 Documents textuels: {len(text_docs)}")
    print(f"🎨 Éléments visuels: {len(visual_elements)}")
    
    # Afficher les éléments visuels
    has_visual = len(visual_elements) > 0
    
    if has_visual:
        await display_visual_elements(visual_elements, user_question)
    
    return text_docs, has_visual

async def display_visual_elements(visual_elements: List[Dict], user_question: str):
    """
    Affiche les éléments visuels dans Chainlit
    """
    if not visual_elements:
        return
    
    # Message d'introduction pour les éléments visuels
    intro_msg = f"📊 **Éléments visuels ANSD trouvés :**\n*{user_question}*\n"
    await cl.Message(content=intro_msg).send()
    
    for i, element in enumerate(visual_elements, 1):
        element_type = element['type']
        metadata = element['metadata']
        
        # Extraire les informations importantes
        caption = metadata.get('caption', f'Élément visuel {i}')
        pdf_name = metadata.get('pdf_name', metadata.get('source_pdf', 'Document ANSD'))
        page = metadata.get('page', metadata.get('page_num', 0))
        
        # Créer le titre de l'élément
        if element_type == 'visual_chart':
            title = f"📊 **Graphique {i}** : {caption}"
        elif element_type == 'visual_table':
            title = f"📋 **Tableau {i}** : {caption}"
        else:
            title = f"📄 **Élément {i}** : {caption}"
        
        # Informations sur la source
        source_info = f"*Source : {pdf_name}"
        if page:
            source_info += f", page {page}"
        source_info += "*"
        
        # Affichage générique avec contenu tronqué
        await cl.Message(
            content=f"{title}\n{source_info}\n\n📝 **Contenu :**\n{element['content'][:500]}..."
        ).send()

# =============================================================================
# FONCTION D'APPEL RAG CORRIGÉE
# =============================================================================

async def call_graph(graph_instance, user_input: str, chat_history: list, graph_type: str, graph_config: dict):
    """Appelle le graphique approprié selon son type avec gestion d'erreurs robuste"""
    
    print(f"🔄 Appel du graphique {graph_type} (type: {graph_config['type']})")
    
    if graph_config["type"] == "class":
        # Pour les classes DirectRAG ou RAGGraph
        print("📞 Appel méthode .ask() sur classe")
        try:
            # Vérifier si la méthode est asynchrone
            import inspect
            if inspect.iscoroutinefunction(graph_instance.ask):
                result = await graph_instance.ask(user_input, chat_history)
            else:
                result = graph_instance.ask(user_input, chat_history)
            print(f"✅ Résultat classe reçu: {type(result)}")
            return result
        except Exception as e:
            print(f"❌ Erreur dans appel classe: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur dans le système RAG: {str(e)}", []
    
    elif graph_config["type"] == "langgraph":
        # Pour les graphiques LangGraph avec gestion d'erreurs améliorée
        print("📞 Appel LangGraph .ainvoke()")
        
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Convertir l'historique en messages
            messages = []
            for user_msg, bot_msg in chat_history:
                messages.append(HumanMessage(content=user_msg))
                messages.append(AIMessage(content=bot_msg))
            
            # Ajouter le message actuel
            messages.append(HumanMessage(content=user_input))
            
            print(f"📝 Messages préparés: {len(messages)} messages")
            
            # Préparer l'input
            graph_input = {"messages": messages}
            
            # Configuration pour Pinecone
            config = {
                "configurable": {
                    "model": "openai/gpt-4o-mini",
                    "retrieval_k": 15,
                    "use_pinecone": True,
                    "pinecone_index": os.getenv('PINECONE_INDEX', 'index-ansd')
                }
            }
            
            # Stratégies d'appel multiples
            result = None
            
            # Stratégie 1: Appel standard
            try:
                print("🔧 Tentative d'appel standard...")
                result = await graph_instance.ainvoke(graph_input, config=config)
                print("✅ Appel standard réussi")
                
            except Exception as e1:
                print(f"⚠️ Erreur appel standard: {e1}")
                
                # Stratégie 2: Sans certaines configurations
                try:
                    print("🔧 Tentative sans configuration complète...")
                    simple_config = {"configurable": {"model": "openai/gpt-4o-mini"}}
                    result = await graph_instance.ainvoke(graph_input, config=simple_config)
                    print("✅ Appel simplifié réussi")
                    
                except Exception as e2:
                    print(f"⚠️ Erreur appel simplifié: {e2}")
                    
                    # Stratégie 3: Sans configuration
                    try:
                        print("🔧 Tentative sans configuration...")
                        result = await graph_instance.ainvoke(graph_input)
                        print("✅ Appel sans config réussi")
                        
                    except Exception as e3:
                        print(f"❌ Toutes les stratégies LangGraph ont échoué")
                        print(f"E1: {e1}")
                        print(f"E2: {e2}")
                        print(f"E3: {e3}")
                        return f"❌ Erreur LangGraph: {e1}", []
            
            # Traitement du résultat
            if result is None:
                print("❌ Aucun résultat obtenu de LangGraph")
                return "❌ Aucune réponse générée par le système", []
            
            print(f"📦 Résultat LangGraph obtenu: {type(result)}")
            
            # Extraire la réponse et les documents
            answer = "Pas de réponse générée"
            sources = []
            
            if isinstance(result, dict):
                # Extraire la réponse des messages
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        answer = last_message.content
                    else:
                        answer = str(last_message)
                    print(f"📝 Réponse extraite: {len(answer)} caractères")
                
                # Extraire les documents sources
                sources = result.get("documents", [])
                print(f"📄 Documents trouvés: {len(sources)}")
            else:
                answer = str(result)
                print(f"📝 Résultat converti en string: {len(answer)} caractères")
            
            return answer, sources
            
        except Exception as e:
            print(f"❌ Erreur globale dans LangGraph: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur système LangGraph: {str(e)}", []
    
    else:
        return f"❌ Type de graphique non supporté: {graph_config['type']}", []

# =============================================================================
# CALLBACKS CHAINLIT
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialisation du chat"""
    
    print("🚀 Initialisation du chat Chainlit...")
    
    if not AVAILABLE_GRAPHS:
        await cl.Message(
            content="""❌ **Aucun système disponible**

**Problèmes possibles :**
• Clé API OpenAI manquante ou invalide
• Clé API Pinecone manquante ou invalide
• Index Pinecone introuvable
• Problème de configuration

**Solutions :**
1. Vérifiez votre fichier `.env` :
   ```
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   PINECONE_INDEX=index-ansd
   ```

2. Vérifiez que votre index Pinecone existe et est accessible

3. Redémarrez l'application"""
        ).send()
        return
    
    # Configuration automatique avec un seul système ou sélection
    if len(AVAILABLE_GRAPHS) == 1:
        graph_key = list(AVAILABLE_GRAPHS.keys())[0]
        graph_info = AVAILABLE_GRAPHS[graph_key]
        
        cl.user_session.set("selected_graph", graph_key)
        
        await cl.Message(
            content=f"""🇸🇳 **Bienvenue dans TERANGA IA - ANSD**

**Assistant Intelligent pour les Statistiques du Sénégal**

✅ **{graph_info['name']}** activé avec Pinecone

📝 *{graph_info['description']}*

📊 **Fonctionnalités :**
• Réponses basées sur les données officielles ANSD
• Recherche dans l'index Pinecone `{os.getenv('PINECONE_INDEX', 'index-ansd')}`
• Affichage automatique des graphiques et tableaux
• Citations précises des sources et pages
• Support des enquêtes : RGPH, EDS, ESPS, EHCVM, ENES

**💡 Exemples de questions :**
• *"Quelle est la population du Sénégal selon le dernier RGPH ?"*
• *"Répartition des ménages par région selon la nature du revêtement du toit"*
• *"Évolution de la population du Sénégal par année"*
• *"Taux de pauvreté par région administrative"*

**🔧 Commandes disponibles :**
• `/help` - Afficher l'aide complète
• `/debug` - Informations de diagnostic
• `/clear` - Effacer l'historique

**🎯 Posez votre question sur les statistiques du Sénégal !**"""
        ).send()
    else:
        # Interface de sélection multiple
        actions = [
            cl.Action(
                name=graph_key,
                value=graph_key,
                label=f"{graph_info['name']}: {graph_info['description']}",
                payload={"graph_type": graph_key}
            )
            for graph_key, graph_info in AVAILABLE_GRAPHS.items()
        ]
        
        await cl.Message(
            content="🤖 **Bienvenue dans le Chatbot RGPH – ANSD**\n\nChoisissez le type de RAG que vous souhaitez utiliser:",
            actions=actions
        ).send()
        
        cl.user_session.set("selected_graph", None)
    
    # Initialiser les variables de session
    cl.user_session.set("chat_history", [])
    print("✅ Session Chainlit initialisée")

# Callbacks pour la sélection des graphiques
@cl.action_callback("simple_rag")
async def on_simple_rag_selected(action):
    await select_graph("simple_rag")

@cl.action_callback("retrieval_graph") 
async def on_retrieval_graph_selected(action):
    await select_graph("retrieval_graph")

@cl.action_callback("self_rag")
async def on_self_rag_selected(action):
    await select_graph("self_rag")

async def select_graph(graph_type: str):
    """Sélectionne le graphique choisi"""
    if graph_type not in AVAILABLE_GRAPHS:
        await cl.Message(
            content=f"❌ Graphique {graph_type} non disponible"
        ).send()
        return
    
    cl.user_session.set("selected_graph", graph_type)
    graph_info = AVAILABLE_GRAPHS[graph_type]
    
    await cl.Message(
        content=f"""✅ **{graph_info['name']}** sélectionné !

📝 *{graph_info['description']}*

🎯 Vous pouvez maintenant poser vos questions sur les données RGPH, EDS, ESPS et autres enquêtes nationales.

Les graphiques et tableaux seront affichés automatiquement quand ils sont pertinents pour votre question."""
    ).send()

@cl.on_message
async def main(message):
    """Traitement principal des messages avec support visuel automatique"""
    
    try:
        print(f"📨 Message reçu: {message.content}")
        
        # Gestion des commandes
        content = message.content.lower().strip()
        
        # Commande de changement de graphique
        if content.startswith("/switch") and len(AVAILABLE_GRAPHS) > 1:
            parts = content.split()
            if len(parts) == 2 and parts[1] in AVAILABLE_GRAPHS:
                await select_graph(parts[1])
            else:
                available = ", ".join(AVAILABLE_GRAPHS.keys())
                await cl.Message(
                    content=f"**Usage:** `/switch [graph_type]`\n\n**Graphiques disponibles:** {available}"
                ).send()
            return
        
        # Commande d'aide
        if content in ["/help", "/aide", "aide", "help"]:
            help_text = f"""🆘 **Aide - Assistant ANSD**

**📊 Types de questions supportées :**
• Statistiques démographiques (population, ménages, etc.)
• Données économiques (PIB, emploi, pauvreté, etc.)
• Indicateurs sociaux (éducation, santé, etc.)
• Répartitions géographiques (par région, milieu, etc.)
• Évolutions temporelles et tendances

**🎨 Affichage automatique :**
• Graphiques : Affichés automatiquement quand pertinents
• Tableaux : Formatés avec données complètes
• Sources : Citations précises avec PDF et page

**💡 Conseils pour de meilleures réponses :**
• Soyez spécifique : "population urbaine Dakar 2023"
• Mentionnez le type de données : "taux de pauvreté", "répartition"
• Précisez la zone : "par région", "milieu rural/urbain"

**🔧 Commandes disponibles :**
• `/help` ou `/aide` : Afficher cette aide
• `/debug` : Informations techniques
• `/clear` : Effacer l'historique"""
            
            if len(AVAILABLE_GRAPHS) > 1:
                help_text += f"\n• `/switch [type]` : Changer de RAG\n\n**Graphiques disponibles :**\n"
                for key, info in AVAILABLE_GRAPHS.items():
                    help_text += f"• `{key}` - {info['name']}: {info['description']}\n"
            
            help_text += f"\n\n**🔌 Configuration Pinecone :**\nIndex actuel : `{os.getenv('PINECONE_INDEX', 'index-ansd')}`"
            
            await cl.Message(content=help_text).send()
            return
        
        # Commande de debug
        if content == "/debug":
            debug_info = f"""🔧 **Informations de Debug**

**🏗️ Configuration :**
• Graphiques disponibles : {list(AVAILABLE_GRAPHS.keys())}
• Graphique actuel : {cl.user_session.get("selected_graph", "Non sélectionné")}
• Support visuel : ✅ Activé

**🔌 Configuration Pinecone :**
• PINECONE_API_KEY : {'✅ Configurée' if os.getenv('PINECONE_API_KEY') else '❌ Manquante'}
• PINECONE_INDEX : {os.getenv('PINECONE_INDEX', 'index-ansd')}
• Index configuré : {'✅ Oui' if os.getenv('PINECONE_INDEX') else '⚠️ Valeur par défaut'}

**🔑 Autres API :**
• OpenAI : {'✅ Configurée' if os.getenv('OPENAI_API_KEY') else '❌ Manquante'}

**📁 Dossiers optionnels :**
• Images : {Path('images').exists() and len(list(Path('images').glob('*.png')))} fichiers
• Tableaux : {Path('tables').exists() and len(list(Path('tables').glob('*.csv')))} fichiers

**💾 Session :**
• Historique : {len(cl.user_session.get('chat_history', []))} échanges"""
            
            await cl.Message(content=debug_info).send()
            return
        
        # Commande de nettoyage
        if content == "/clear":
            cl.user_session.set("chat_history", [])
            await cl.Message(content="🧹 **Historique effacé**\n\nL'historique de conversation a été remis à zéro.").send()
            return
        
        # Vérifier qu'un graphique a été sélectionné
        selected_graph = cl.user_session.get("selected_graph")
        
        if not selected_graph:
            await cl.Message(
                content="⚠️ **Veuillez d'abord sélectionner un type de RAG**\n\nUtilisez les boutons ci-dessus ou tapez `/help` pour voir les commandes disponibles."
            ).send()
            return
        
        if selected_graph not in AVAILABLE_GRAPHS:
            await cl.Message(
                content=f"❌ Graphique {selected_graph} non disponible\n\nUtilisez `/switch [type]` pour changer de graphique."
            ).send()
            return
        
        # Récupérer l'historique
        chat_history = cl.user_session.get("chat_history", [])
        
        # Extraire le texte du message
        user_input = message.content
        
        print(f"❓ Question: {user_input}")
        print(f"📚 Historique: {len(chat_history)} échanges")
        
        # Limiter l'historique envoyé (garder les 10 derniers échanges)
        short_history = chat_history[-10:]
        
        # Afficher un indicateur de traitement
        graph_info = AVAILABLE_GRAPHS[selected_graph]
        processing_msg = await cl.Message(
            content=f"🔍 **Recherche dans Pinecone ({os.getenv('PINECONE_INDEX', 'index-ansd')})...**\n\n*Analyse en cours avec {graph_info['name']} et détection automatique du contenu visuel*"
        ).send()
        
        # Appeler le graphique approprié
        print(f"🔄 Appel du graphique {selected_graph}...")
        answer, documents = await call_graph(
            graph_info["instance"], 
            user_input, 
            short_history, 
            selected_graph,
            graph_info
        )
        
        print(f"✅ Réponse reçue: {len(answer) if answer else 0} chars, {len(documents) if documents else 0} docs")
        
        # Supprimer le message de traitement
        await processing_msg.remove()
        
        # Détecter et afficher automatiquement le contenu visuel
        has_visual = False
        if documents:
            print(f"📄 {len(documents)} documents récupérés")
            try:
                text_docs, has_visual = await detect_and_display_visual_content(documents, user_input)
                print(f"🎨 Contenu visuel détecté: {has_visual}")
                
                if has_visual:
                    # Ajouter une note à la réponse
                    answer += "\n\n*📊 Les éléments visuels correspondants sont affichés ci-dessus.*"
            except Exception as e:
                print(f"⚠️ Erreur détection visuel: {e}")
        
        # Mettre à jour l'historique
        chat_history.append((user_input, answer))
        
        # Limiter la taille de l'historique
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        cl.user_session.set("chat_history", chat_history)
        
        # Envoyer la réponse
        await cl.Message(content=answer).send()
        
        print(f"✅ Réponse envoyée. Historique: {len(chat_history)} échanges")
        
    except Exception as e:
        print(f"❌ Erreur dans main: {e}")
        import traceback
        traceback.print_exc()
        
        await cl.Message(
            content=f"""❌ **Erreur technique**

Une erreur s'est produite lors du traitement de votre question.

**Détails:** {str(e)}

**Solutions suggérées:**
• Vérifiez vos clés API (OpenAI, Pinecone)
• Vérifiez que votre index Pinecone est accessible
• Redémarrez l'application avec `chainlit run chainlit_app.py`
• Tapez `/debug` pour plus d'informations
• Reformulez votre question

Veuillez réessayer."""
        ).send()

# =============================================================================
# DÉMARRAGE
# =============================================================================

if __name__ == "__main__":
    print("🚀 Lancement de TERANGA IA - ANSD")
    print("📊 Système RAG avec Pinecone et support visuel")
    print("🔗 Interface Chainlit prête")
    print(f"📈 Graphiques disponibles: {list(AVAILABLE_GRAPHS.keys())}")
    print(f"🔌 Index Pinecone: {os.getenv('PINECONE_INDEX', 'index-ansd')}")
    
    if not AVAILABLE_GRAPHS:
        print("⚠️ ATTENTION: Aucun graphique disponible!")
        print("Vérifiez vos imports et votre configuration Pinecone.")
    
    # L'application sera lancée avec: chainlit run chainlit_app.py