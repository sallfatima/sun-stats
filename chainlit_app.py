import chainlit as cl
import sys
import os

# Ajouter le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import direct du Simple RAG
try:
    from simple_rag.graph import graph as simple_rag_graph
    RAG_AVAILABLE = True
    print("✅ Simple RAG chargé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import Simple RAG: {e}")
    RAG_AVAILABLE = False

async def call_simple_rag(user_input: str, chat_history: list):
    """Appelle le Simple RAG avec le message utilisateur"""
    
    if not RAG_AVAILABLE:
        return "❌ Simple RAG non disponible", []
    
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Convertir l'historique en messages LangChain
        messages = []
        for user_msg, bot_msg in chat_history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        # Configuration par défaut
        config = None
        
        print(f"🔍 Appel Simple RAG avec {len(messages)} messages")
        
        # Appeler le graphique Simple RAG sans configuration spécifique
        result = await simple_rag_graph.ainvoke({"messages": messages}, config=config)
        
        # Extraire la réponse
        if "messages" in result and result["messages"]:
            answer = result["messages"][-1].content
            print(f"✅ Réponse générée: {len(answer)} caractères")
        else:
            answer = "❌ Aucune réponse générée par Simple RAG"
        
        # Extraire les documents sources
        sources = result.get("documents", [])
        print(f"📄 Documents récupérés: {len(sources)}")
        
        return answer, sources
        
    except Exception as e:
        print(f"❌ Erreur Simple RAG: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Erreur technique: {str(e)}", []

@cl.on_chat_start
async def on_chat_start():
    """Initialisation du chat Simple RAG"""
    
    if not RAG_AVAILABLE:
        await cl.Message(
            content="❌ **Simple RAG non disponible**\n\n"
                   "Vérifiez que le module simple_rag est correctement installé."
        ).send()
        return
    
    # Message de bienvenue
    welcome_message = """🇸🇳 **Bienvenue dans Sunu-Stats - ANSD**

**Assistant Intelligent pour les Statistiques du Sénégal**


📊 **Données disponibles :**
• **RGPH** - Recensement Général de la Population et de l'Habitat
• **EDS** - Enquête Démographique et de Santé  
• **ESPS/EHCVM** - Enquêtes sur la Pauvreté et Conditions de Vie
• **ENES** - Enquête Nationale sur l'Emploi
• **Comptes Nationaux** - Données économiques

💡 **Exemples de questions :**
• Quelle est la population du Sénégal selon le dernier RGPH ?
• Quel est le taux de pauvreté au Sénégal ?
• Comment évolue le taux d'alphabétisation ?
• Quels sont les indicateurs de santé maternelle ?

🆘 **Aide :** Tapez `/help` pour plus d'informations

Posez vos questions sur les statistiques et enquêtes nationales !"""

    await cl.Message(content=welcome_message).send()
    
    # Initialiser l'historique du chat
    cl.user_session.set("chat_history", [])

@cl.on_message
async def main(message):
    """Traitement principal des messages avec Simple RAG"""
    
    content = message.content.strip()
    
    # Gestion des commandes spéciales
    if content.lower() == "/help":
        help_text = """**🆘 Aide Sunu-Stats - ANSD**

**📋 Commandes disponibles :**
• `/help` - Afficher cette aide
• `/clear` - Effacer l'historique de conversation

**📊 Types de données disponibles :**
• **Démographiques** - Population, natalité, mortalité
• **Économiques** - PIB, pauvreté, emploi, croissance
• **Sociales** - Éducation, santé, alphabétisation
• **Géographiques** - Régions, départements, communes

**🎯 Types d'enquêtes ANSD :**
• **RGPH** - Recensement (données population/habitat)
• **EDS** - Enquête Démographique et Santé
• **ESPS** - Enquête Suivi Pauvreté Sénégal
• **EHCVM** - Enquête Conditions Vie Ménages
• **ENES** - Enquête Nationale Emploi Sénégal

**💡 Conseils pour de meilleures réponses :**
• Soyez spécifique dans vos questions
• Mentionnez l'année si important
• Précisez la région si nécessaire
• Demandez des sources précises

**🔧 Système :** Simple RAG avec base documentaire ANSD"""

        await cl.Message(content=help_text).send()
        return
    
    if content.lower() == "/clear":
        cl.user_session.set("chat_history", [])
        await cl.Message(
            content="🧹 **Historique effacé**\n\nVous pouvez recommencer une nouvelle conversation."
        ).send()
        return
    
    # Vérifier que Simple RAG est disponible
    if not RAG_AVAILABLE:
        await cl.Message(
            content="❌ Simple RAG non disponible. Redémarrez l'application."
        ).send()
        return
    
    # Traitement du message principal
    try:
        # Récupérer l'historique
        chat_history = cl.user_session.get("chat_history", [])
        
        # Limiter l'historique pour éviter de surcharger
        short_history = chat_history[-5:]  # Garder les 5 derniers échanges
        
        # Afficher indicateur de traitement
        processing_msg = await cl.Message(
            content="🔍 **Recherche en cours...**\n\n"
                   "• Récupération des documents ANSD\n"
                   "• Analyse des données statistiques\n"
                   "• Génération de la réponse..."
        ).send()
        
        # Appeler Simple RAG
        answer, sources = await call_simple_rag(content, short_history)
        
        # Supprimer le message de traitement
        await processing_msg.remove()
        
        # Mettre à jour l'historique
        chat_history.append((content, answer))
        cl.user_session.set("chat_history", chat_history)
        
        # Préparer la réponse finale
        response_content = f"**📊 Sunu-Stats - ANSD répond :**\n\n{answer}"
        
        # Ajouter informations sur les sources si disponibles
        if sources and len(sources) > 0:
            response_content += f"\n\n📚 **Sources consultées :** {len(sources)} document(s) ANSD"
        
        # Envoyer la réponse
        await cl.Message(content=response_content).send()
        
        # Optionnel: Afficher détails des sources pour debug
        if sources and len(sources) > 0:
            sources_text = f"📄 **Détails des sources :**\n\n"
            for i, doc in enumerate(sources[:3], 1):  # Limiter à 3 sources
                if hasattr(doc, 'metadata') and doc.metadata:
                    pdf_name = doc.metadata.get('pdf_name', 'Document ANSD')
                    page_num = doc.metadata.get('page_num', 'N/A')
                    if '/' in pdf_name:
                        pdf_name = pdf_name.split('/')[-1]
                    sources_text += f"• **Source {i}:** {pdf_name}"
                    if page_num != 'N/A':
                        sources_text += f" (page {page_num})"
                    sources_text += "\n"
                else:
                    sources_text += f"• **Source {i}:** Document ANSD\n"
            
            # Envoyer les détails des sources (optionnel, décommentez si souhaité)
            # await cl.Message(content=sources_text).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Erreur lors du traitement**\n\n"
                   f"Une erreur technique s'est produite :\n"
                   f"`{str(e)}`\n\n"
                   f"Veuillez réessayer ou reformuler votre question."
        ).send()
        
        print(f"❌ Erreur détaillée: {e}")
        import traceback
        traceback.print_exc()

# Configuration optionnelle pour le débogage
if __name__ == "__main__":
    print("🚀 Démarrage Sunu-Stats - ANSD (Simple RAG)")
    print(f"📊 Simple RAG disponible: {RAG_AVAILABLE}")
    
    if RAG_AVAILABLE:
        print("✅ Prêt à répondre aux questions sur les statistiques du Sénégal")
    else:
        print("❌ Vérifiez l'installation du module simple_rag")