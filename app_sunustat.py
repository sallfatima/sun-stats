# app_sunustat_working.py - Client Chainlit qui fonctionne avec votre backend
import os
import chainlit as cl
import httpx
import json
import asyncio
from typing import Dict, Optional

# Configuration
BACKEND_URL = os.environ.get("LANGGRAPH_DEPLOYMENT", "http://localhost:8001")

# Configuration OAuth optionnelle
if os.environ.get("DATABASE_URL"):
    @cl.oauth_callback
    def oauth_callback(
      provider_id: str,
      token: str,
      raw_user_data: Dict[str, str],
      default_user: cl.User,
    ) -> Optional[cl.User]:
      return default_user

async def test_backend_connection():
    """Teste la connexion au backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return True, data
            else:
                return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

async def get_assistant_info():
    """Récupère les informations de l'assistant"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/assistants/search", timeout=5.0)
            if response.status_code == 200:
                assistants = response.json()
                if assistants:
                    return assistants[0]
            return None
    except Exception as e:
        print(f"Erreur récupération assistant: {e}")
        return None

async def create_thread():
    """Crée un nouveau thread"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BACKEND_URL}/threads", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        print(f"Erreur création thread: {e}")
        return None

async def stream_chat_response(thread_id: str, message_content: str):
    """Stream la réponse du chat"""
    try:
        payload = {
            "messages": [{"content": message_content, "type": "human"}],
            "config": {"search_kwargs": {"k": 10}}
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", 
                f"{BACKEND_URL}/threads/{thread_id}/runs/stream",
                json=payload,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code != 200:
                    yield f"❌ Erreur backend: {response.status_code}"
                    return
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Parser les événements SSE
                            if line.startswith("data: "):
                                data_str = line[6:]  # Enlever "data: "
                                if data_str.strip() == "[DONE]":
                                    break
                                
                                event_data = json.loads(data_str)
                                
                                # Traiter les événements de streaming
                                if event_data.get("event") == "events":
                                    inner_data = event_data.get("data", {})
                                    event_type = inner_data.get("event", "")
                                    
                                    if event_type == "on_chat_model_stream":
                                        chunk_data = inner_data.get("data", {})
                                        chunk = chunk_data.get("chunk", {})
                                        content = chunk.get("content", "")
                                        if content:
                                            yield content
                                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Erreur parsing événement: {e}")
                            continue
                            
    except Exception as e:
        yield f"❌ Erreur de streaming: {str(e)}"

@cl.on_chat_start
async def on_chat_start():
    """Initialisation du chat"""
    
    # Tester la connexion backend
    connected, info = await test_backend_connection()
    
    if not connected:
        await cl.Message(
            content=f"❌ **Backend SunuStat non accessible**\n\n"
                   f"URL: `{BACKEND_URL}`\n"
                   f"Erreur: {info}\n\n"
                   f"Vérifiez que le backend est démarré :\n"
                   f"```bash\n"
                   f"python backend_server_sunustat.py\n"
                   f"```"
        ).send()
        cl.user_session.set("backend_connected", False)
        return
    
    # Récupérer les infos de l'assistant
    assistant = await get_assistant_info()
    if not assistant:
        await cl.Message(
            content="⚠️ **Assistant non trouvé**\n\nLe backend répond mais l'assistant n'est pas configuré."
        ).send()
        cl.user_session.set("backend_connected", False)
        return
    
    # Créer un thread
    thread = await create_thread()
    if not thread:
        await cl.Message(
            content="⚠️ **Impossible de créer un thread**\n\nErreur de communication avec le backend."
        ).send()
        cl.user_session.set("backend_connected", False)
        return
    
    # Stocker les informations en session
    cl.user_session.set("backend_connected", True)
    cl.user_session.set("thread_id", thread["thread_id"])
    cl.user_session.set("assistant", assistant)
    
    # Message de bienvenue avec informations du backend
    rag_status = "✅ Activé" if info.get("rag_available", False) else "⚠️ Mode démo"
    
    welcome_message = f"""🇸🇳 **Bienvenue dans SunuStat - ANSD**

**Assistant Intelligent pour les Statistiques du Sénégal**

**🔧 Statut du système :**
• Backend : ✅ Connecté ({BACKEND_URL})
• Simple RAG : {rag_status}
• Version : {info.get('version', 'N/A')}

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

@cl.on_message
async def main(message: cl.Message):
    """Traitement des messages"""
    
    # Vérifier la connexion backend
    if not cl.user_session.get("backend_connected", False):
        await cl.Message(
            content="❌ **Backend non connecté**\n\nRafraîchissez la page pour reconnecter."
        ).send()
        return
    
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        await cl.Message(
            content="❌ **Session invalide**\n\nRafraîchissez la page."
        ).send()
        return
    
    # Préparer le streaming
    msg = cl.Message(content="")
    
    try:
        # Afficher l'étape de traitement
        async with cl.Step(name="🔍 Analyse ANSD") as step:
            step.input = f"Question: {message.content}"
            step.output = "Recherche en cours dans les documents ANSD..."
            
            # Démarrer le streaming
            response_started = False
            
            async for chunk in stream_chat_response(thread_id, message.content):
                if chunk.strip():
                    if not response_started:
                        step.output = "✅ Réponse reçue du backend"
                        response_started = True
                    
                    await msg.stream_token(chunk)
        
        # Finaliser le message
        await msg.send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Erreur de traitement**\n\n"
                   f"Erreur: {str(e)}\n\n"
                   f"Vérifiez les logs du backend."
        ).send()
        print(f"❌ Erreur client: {e}")

# Point d'entrée pour le débogage
if __name__ == "__main__":
    print("🚀 Client SunuStat Chainlit")
    print(f"🔗 Backend: {BACKEND_URL}")
    
    # Test rapide de connexion
    import asyncio
    
    async def quick_test():
        connected, info = await test_backend_connection()
        if connected:
            print("✅ Backend accessible")
            print(f"   RAG: {'Disponible' if info.get('rag_available') else 'Mode démo'}")
        else:
            print(f"❌ Backend non accessible: {info}")
    
    asyncio.run(quick_test())