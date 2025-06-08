import chainlit as cl
from src.simple_rag.graph import RAGGraph

chatbot = RAGGraph()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])

@cl.on_message
async def main(message):
    # 1. Récupérer l'historique
    chat_history = cl.user_session.get("chat_history", [])

    # 2. Extraire le texte de l'objet Message
    user_input = message.content

    # 3. Limiter l'historique envoyé
    short_history = chat_history[-5:]

    # 4. Appeler le chatbot
    answer, sources = chatbot.ask(user_input, short_history)

    # 5. Mettre à jour l'historique dans la session
    chat_history.append((user_input, answer))
    cl.user_session.set("chat_history", chat_history)

    # 6. Envoyer la réponse
    await cl.Message(content=answer).send()

    # 7. Envoyer la liste des sources
    if sources:
        sources_md = "\n".join(
            f"- `{doc.metadata.get('source_pdf','inconnu')}` (chunk `{doc.metadata.get('chunk','?')}`)"
            for doc in sources
        )
        await cl.Message(content=f"**Sources utilisées :**\n{sources_md}").send()
