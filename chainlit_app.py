import chainlit as cl
import sys
import os

# Ajouter le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports dynamiques pour éviter les erreurs d'import
def safe_import_rag_graph():
    """Import sécurisé de RAGGraph"""
    try:
        from simple_rag.graph import graph as simple_rag_graph
        return simple_rag_graph, "langgraph"
    except ImportError:
        try:
            # Fallback vers une classe RAGGraph si elle existe
            from simple_rag.graph import RAGGraph
            return RAGGraph(), "class"
        except ImportError as e:
            print(f"❌ Erreur d'import: {e}")
            return None, None

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
        "description": "RAG basique avec récupération et génération",
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

async def call_graph(graph_instance, user_input: str, chat_history: list, graph_type: str, graph_config: dict):
    """Appelle le graphique approprié selon son type"""
    
    if graph_config["type"] == "class":
        # Pour RAGGraph avec méthode ask
        return graph_instance.ask(user_input, chat_history)
    
    elif graph_config["type"] == "langgraph":
        # Pour les graphiques LangGraph
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Convertir l'historique en messages
        messages = []
        for user_msg, bot_msg in chat_history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        # Appeler le graphique avec configuration par défaut
        config = {"configurable": {"model": "openai/gpt-4o"}}
        
        try:
            result = await graph_instance.ainvoke({"messages": messages}, config=config)
        except Exception as e:
            # Fallback sans config si erreur
            try:
                result = await graph_instance.ainvoke({"messages": messages})
            except Exception as e2:
                return f"Erreur lors de l'appel au graphique: {e2}", []
        
        # Extraire la réponse et les documents
        if "messages" in result and result["messages"]:
            answer = result["messages"][-1].content
        else:
            answer = "Pas de réponse générée"
            
        sources = result.get("documents", [])
        
        return answer, sources
    
    else:
        return "Type de graphique non supporté", []

@cl.on_chat_start
async def on_chat_start():
    """Initialisation du chat"""
    
    if not AVAILABLE_GRAPHS:
        await cl.Message(
            content="❌ **Aucun graphique disponible**\n\nVérifiez que les modules sont correctement installés."
        ).send()
        return
    
    # Interface de sélection du graphique
    if len(AVAILABLE_GRAPHS) > 1:
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
    else:
        # Un seul graphique disponible, le sélectionner automatiquement
        graph_key = list(AVAILABLE_GRAPHS.keys())[0]
        graph_info = AVAILABLE_GRAPHS[graph_key]
        
        cl.user_session.set("selected_graph", graph_key)
        
        await cl.Message(
            content="🇸🇳 **Bienvenue dans TERANGA IA - ANSD**\n\n"
                   f"Assistant Intelligent pour les Statistiques du Sénégal\n\n"
                   f"✅ **{graph_info['name']}** activé\n\n"
                   f"📝 *{graph_info['description']}*\n\n"
                   f"**Exemples de questions :**\n"
                   f"• Quelle est la population du Sénégal selon le dernier RGPH ?\n"
                   f"• Quel est le taux de pauvreté au Sénégal ?\n"
                   f"• Comment évolue le taux d'alphabétisation ?\n\n"
                   f"Posez vos questions sur les statistiques et enquêtes nationales !"
        ).send()
    
    # Initialiser les variables de session
    cl.user_session.set("chat_history", [])

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
        content=f"✅ **{graph_info['name']}** sélectionné !\n\n"
               f"📝 *{graph_info['description']}*\n\n"
               f"Vous pouvez maintenant poser vos questions sur les données RGPH."
    ).send()

@cl.on_message
async def main(message):
    """Traitement principal des messages"""
    
    # Gestion des commandes
    content = message.content.lower().strip()
    
    if content.startswith("/switch"):
        parts = content.split()
        if len(parts) == 2 and parts[1] in AVAILABLE_GRAPHS:
            await select_graph(parts[1])
        else:
            available = ", ".join(AVAILABLE_GRAPHS.keys())
            await cl.Message(
                content=f"Usage: /switch [graph_type]\nGraphiques disponibles: {available}"
            ).send()
        return
    
    if content == "/help":
        help_text = "**🆘 Commandes disponibles:**\n\n"
        help_text += "• `/switch [graph_type]` - Changer de graphique\n"
        help_text += "• `/help` - Afficher cette aide\n\n"
        help_text += "**📊 Enquêtes et données disponibles :**\n"
        help_text += "• **RGPH** - Recensement Général Population & Habitat\n"
        help_text += "• **EDS** - Enquête Démographique et de Santé\n"
        help_text += "• **ESPS/EHCVM** - Enquêtes sur la Pauvreté\n"
        help_text += "• **ENES** - Enquête Nationale sur l'Emploi\n"
        help_text += "• **Comptes Nationaux** - Données économiques\n\n"
        for key, info in AVAILABLE_GRAPHS.items():
            help_text += f"• `{key}` - {info['name']}: {info['description']}\n"
        
        await cl.Message(content=help_text).send()
        return
    
    # Vérifier qu'un graphique a été sélectionné
    selected_graph = cl.user_session.get("selected_graph")
    
    if not selected_graph:
        await cl.Message(
            content="⚠️ Veuillez d'abord sélectionner un type de RAG."
        ).send()
        return
    
    if selected_graph not in AVAILABLE_GRAPHS:
        await cl.Message(
            content=f"❌ Graphique {selected_graph} non disponible"
        ).send()
        return
    
    try:
        # 1. Récupérer l'historique
        chat_history = cl.user_session.get("chat_history", [])
        
        # 2. Extraire le texte du message
        user_input = message.content
        
        # 3. Limiter l'historique envoyé
        short_history = chat_history[-5:]
        
        # 4. Afficher un indicateur de traitement
        graph_info = AVAILABLE_GRAPHS[selected_graph]
        processing_msg = await cl.Message(
            content=f"🔍 Traitement avec **{graph_info['name']}**..."
        ).send()
        
        # 5. Appeler le graphique approprié
        answer, sources = await call_graph(
            graph_info["instance"], 
            user_input, 
            short_history, 
            selected_graph,
            graph_info
        )
        
        # 6. Supprimer le message de traitement
        await processing_msg.remove()
        
        # 7. Mettre à jour l'historique
        chat_history.append((user_input, answer))
        cl.user_session.set("chat_history", chat_history)
        
        # 8. Envoyer la réponse
        await cl.Message(
            content=f"**{graph_info['name']}** répond:\n\n{answer}"
        ).send()

        
    except Exception as e:
        await cl.Message(
            content=f"❌ Erreur lors du traitement: {str(e)}"
        ).send()
        print(f"Erreur détaillée: {e}")
        import traceback
        traceback.print_exc()