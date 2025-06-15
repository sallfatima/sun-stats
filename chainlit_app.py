import chainlit as cl
import sys
import os
from datetime import datetime

# ------------------------------------------------------------
# Configuration des chemins et import du Simple RAG
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

try:
    from simple_rag.graph import graph as simple_rag_graph
    RAG_AVAILABLE = True
    print("✅ Simple RAG chargé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import Simple RAG: {e}")
    RAG_AVAILABLE = False

# Fichier pour enregistrer les feedbacks locaux (👍 / 👎)
FEEDBACK_LOG = os.path.join(BASE_DIR, "feedback.log")

# ------------------------------------------------------------
# Fonction d'appel au Simple RAG
# ------------------------------------------------------------
async def call_simple_rag(user_input: str, chat_history: list):
    """Appelle Simple RAG avec l'entrée utilisateur + historique."""

    if not RAG_AVAILABLE:
        return "❌ Simple RAG non disponible", []

    try:
        from langchain_core.messages import HumanMessage, AIMessage

        # Reconstruction de l'historique pour LangChain
        messages = []
        for user_msg, bot_msg in chat_history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        messages.append(HumanMessage(content=user_input))

        result = await simple_rag_graph.ainvoke({"messages": messages}, config=None)

        # Réponse générée
        answer = "❌ Aucune réponse générée par Simple RAG"
        if result.get("messages"):
            answer = result["messages"][-1].content

        # Documents sources (liste de Document)
        sources = result.get("documents", [])
        return answer, sources

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Erreur technique: {e}", []

# ------------------------------------------------------------
# Utilitaires feedback 👍 / 👎
# ------------------------------------------------------------

def _log_feedback(mid: str, value: str):
    """Enregistre le feedback dans un fichier CSV minimal."""
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()},{mid},{value}\n")
    except Exception as log_err:
        print("Erreur log feedback:", log_err)

@cl.action_callback("like")
async def on_like(action: cl.Action):
    mid = action.payload.get("mid")
    print(f"👍 Like reçu pour {mid}")
    _log_feedback(mid, "like")
    await action.remove()

@cl.action_callback("dislike")
async def on_dislike(action: cl.Action):
    mid = action.payload.get("mid")
    print(f"👎 Dislike reçu pour {mid}")
    _log_feedback(mid, "dislike")
    await action.remove()

# ------------------------------------------------------------
# Cycle de vie du chat
# ------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Message de bienvenue + initialisation."""
    if not RAG_AVAILABLE:
        await cl.Message(
            content="❌ **Simple RAG non disponible**\n\nVérifiez l'installation du module `simple_rag`."
        ).send()
        return

    welcome_message = """🇸🇳 **Bienvenue dans SunuStat - ANSD**

**Assistant Intelligent pour les Statistiques du Sénégal**

Les réponses s'appuient exclusivement sur les publications officielles de l'Agence Nationale de la Statistique et de la Démographie du Sénégal (ANSD).

📊 **Données utilisées :**
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

🆘 **Aide :** Tapez `/Aide` pour plus d'informations

Posez vos questions sur les statistiques et enquêtes nationales !"""

    await cl.Message(content=welcome_message).send()
    cl.user_session.set("chat_history", [])

@cl.on_message
async def main(message: cl.Message):
    content = message.content.strip()

    # --------------------------------------------------------
    # Commandes spéciales
    # --------------------------------------------------------
    if content.lower() == "/Aide":
        help_text = """**🆘 Aide SunuStat - ANSD**

**📋 Commandes disponibles :**
• `/Aide` - Afficher cette aide
• `/Efface` - Effacer l'historique de conversation

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

    if content.lower() == "/Effacer":
        cl.user_session.set("chat_history", [])
        await cl.Message(content="🧹 **Historique effacé**\n\nVous pouvez recommencer une nouvelle conversation.").send()
        return

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------
    if not RAG_AVAILABLE:
        await cl.Message(content="❌ Simple RAG non disponible. Redémarrez l'application.").send()
        return

    # --------------------------------------------------------
    # Traitement principal
    # --------------------------------------------------------
    chat_history = cl.user_session.get("chat_history", [])
    short_history = chat_history[-5:]  # Limiter l'historique pour ne pas surcharger

    processing = await cl.Message(content="🔍 **Recherche en cours...**\n\n• Récupération des documents ANSD\n• Analyse des données statistiques\n• Génération de la réponse...").send()

    answer, sources = await call_simple_rag(content, short_history)
    await processing.remove()

    chat_history.append((content, answer))
    cl.user_session.set("chat_history", chat_history)

    response_content = f"**📊 SunuStat - ANSD répond :**\n\n{answer}"
    if sources:
        response_content += f"\n\n📚 **Sources consultées :** {len(sources)} document(s) ANSD"

    # 1️⃣ Envoi de la réponse
    assistant_msg = cl.Message(content=response_content)
    await assistant_msg.send()

    # 2️⃣ Boutons feedback Oui / Non
    feedback_actions = [
        cl.Action(
            name="like",
            label="Oui",
            icon="thumbs-up",
            tooltip="Réponse utile",
            payload={"mid": assistant_msg.id},
        ),
        cl.Action(
            name="dislike",
            label="Non",
            icon="thumbs-down",
            tooltip="Réponse à améliorer",
            payload={"mid": assistant_msg.id},
        ),
    ]

    await cl.Message(content="Qualité de la réponse ?", actions=feedback_actions).send()

    # --------------------------------------------------------
    # (Optionnel) Affichage détaillé des sources
    # --------------------------------------------------------
    # if sources:
    #     details = "📄 **Détails des sources :**\n\n" + "\n".join(
    #         f"• {doc.metadata.get('pdf_name', 'Document ANSD')} (page {doc.metadata.get('page_num', 'N/A')})"
    #         for doc in sources[:3]
    #     )
    #     await cl.Message(content=details).send()

# ------------------------------------------------------------
# Exécution directe (debug)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Démarrage SunuStat - ANSD (Simple RAG)")
    print("📊 Simple RAG disponible:", RAG_AVAILABLE)
    if RAG_AVAILABLE:
        print("✅ Prêt à répondre aux questions sur les statistiques du Sénégal")
    else:
        print("❌ Vérifiez l'installation du module simple_rag")
