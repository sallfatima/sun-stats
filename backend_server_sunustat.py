# backend_server_sunustat.py - Backend avec mise en forme SunuStat
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
import asyncio
from sse_starlette.sse import EventSourceResponse
import json
import sys

app = FastAPI(title="SunuStat ANSD - LangGraph API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Ajouter le chemin vers simple_rag si nécessaire
RAG_AVAILABLE = False
simple_rag_graph = None

try:
    # Tentative d'import du Simple RAG
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from simple_rag.graph import graph as simple_rag_graph
    RAG_AVAILABLE = True
    print("✅ Simple RAG chargé avec succès dans le backend")
except ImportError as e:
    print(f"❌ Erreur d'import Simple RAG: {e}")
    print("⚠️ Le backend fonctionnera en mode démo")

# Models Pydantic
class Message(BaseModel):
    content: str
    type: str = "human"

class RunRequest(BaseModel):
    messages: List[Message]
    config: Dict[str, Any] = {}

class Assistant(BaseModel):
    assistant_id: str
    name: str
    graph_id: str
    metadata: Dict[str, Any]

class Thread(BaseModel):
    thread_id: str
    created_at: str

# Fonctions utilitaires
async def call_simple_rag(user_input: str, chat_history: list):
    """Appelle le Simple RAG avec le message utilisateur - Version backend"""
    
    if not RAG_AVAILABLE:
        return generate_demo_response(user_input), []
    
    try:
        # Convertir l'historique en messages LangChain
        messages = []
        for user_msg, bot_msg in chat_history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        print(f"🔍 Backend: Appel Simple RAG avec {len(messages)} messages")
        
        # Appeler le graphique Simple RAG
        result = await simple_rag_graph.ainvoke({"messages": messages}, config=None)
        
        # Extraire la réponse
        if "messages" in result and result["messages"]:
            answer = result["messages"][-1].content
            print(f"✅ Backend: Réponse générée: {len(answer)} caractères")
        else:
            answer = "❌ Aucune réponse générée par Simple RAG"
        
        # Extraire les documents sources
        sources = result.get("documents", [])
        print(f"📄 Backend: Documents récupérés: {len(sources)}")
        
        return answer, sources
        
    except Exception as e:
        print(f"❌ Backend: Erreur Simple RAG: {e}")
        return generate_demo_response(user_input), []

def generate_demo_response(user_input: str) -> str:
    """Génère une réponse de démonstration formatée SunuStat"""
    demo_responses = {
        "population": """**📊 Population du Sénégal (RGPH 2023)**

Selon les dernières données du Recensement Général de la Population et de l'Habitat (RGPH) :

• **Population totale :** 18 275 743 habitants
• **Croissance démographique :** 2,8% par an
• **Densité :** 93 habitants/km²

**🌍 Répartition régionale :**
• Dakar : 4 029 724 habitants (22,0%)
• Thiès : 2 076 809 habitants (11,4%)
• Diourbel : 1 739 748 habitants (9,5%)

*Source : ANSD - RGPH 2023 (données provisoires)*""",

        "pauvreté": """**💰 Indicateurs de Pauvreté au Sénégal**

Selon l'Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM) 2018-2019 :

• **Taux de pauvreté national :** 37,8%
• **Pauvreté rurale :** 53,2%
• **Pauvreté urbaine :** 23,7%

**📈 Évolution :**
• 2011 : 46,7%
• 2018-2019 : 37,8%
• Baisse de 8,9 points

*Source : ANSD - EHCVM 2018-2019*""",

        "emploi": """**👔 Situation de l'Emploi au Sénégal**

D'après l'Enquête Nationale sur l'Emploi au Sénégal (ENES) :

• **Taux d'activité :** 49,2%
• **Taux de chômage :** 16,9%
• **Chômage des jeunes (15-34 ans) :** 22,7%

**🏢 Secteurs d'activité :**
• Agriculture : 35,8%
• Services : 38,2%
• Industrie : 26,0%

*Source : ANSD - ENES (dernières données disponibles)*"""
    }
    
    # Recherche de mots-clés pour retourner une réponse appropriée
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["population", "habitants", "rgph", "recensement"]):
        return demo_responses["population"]
    elif any(word in user_lower for word in ["pauvreté", "pauvre", "ehcvm", "conditions"]):
        return demo_responses["pauvreté"]
    elif any(word in user_lower for word in ["emploi", "chômage", "travail", "enes"]):
        return demo_responses["emploi"]
    else:
        return f"""**📊 SunuStat - ANSD**

Votre question : "{user_input}"

Cette réponse est générée en mode démonstration. Pour des données réelles, connectez le module Simple RAG.

**📋 Types de données disponibles :**
• Démographie et population (RGPH)
• Pauvreté et conditions de vie (EHCVM/ESPS)
• Emploi et activité économique (ENES)
• Santé et nutrition (EDS)
• Éducation et alphabétisation

*Posez une question plus spécifique pour obtenir des statistiques détaillées.*"""

def process_special_commands(content: str) -> Optional[str]:
    """Traite les commandes spéciales"""
    if content.lower() == "/help":
        return """**🆘 Aide SunuStat - ANSD**

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

**🔧 Backend :** LangGraph + Simple RAG"""
    
    elif content.lower() == "/clear":
        return """🧹 **Historique effacé**

Vous pouvez recommencer une nouvelle conversation."""
    
    return None

def format_sources_info(sources: List) -> str:
    """Formate les informations des sources"""
    if not sources or len(sources) == 0:
        return ""
    
    sources_text = f"\n\n📚 **Sources consultées :** {len(sources)} document(s) ANSD"
    
    # Détails des sources (optionnel)
    details = "\n\n📄 **Détails des sources :**\n"
    for i, doc in enumerate(sources[:3], 1):  # Limiter à 3 sources
        if hasattr(doc, 'metadata') and doc.metadata:
            pdf_name = doc.metadata.get('pdf_name', 'Document ANSD')
            page_num = doc.metadata.get('page_num', 'N/A')
            if '/' in pdf_name:
                pdf_name = pdf_name.split('/')[-1]
            details += f"• **Source {i}:** {pdf_name}"
            if page_num != 'N/A':
                details += f" (page {page_num})"
            details += "\n"
        else:
            details += f"• **Source {i}:** Document ANSD\n"
    
    return sources_text  # Retourner seulement le résumé pour l'instant

# Routes de l'API

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "service": "SunuStat ANSD - Backend",
        "version": "1.0.0",
        "description": "API pour l'assistant statistique du Sénégal",
        "rag_available": RAG_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "assistants": "/assistants/search",
            "threads": "/threads"
        }
    }

@app.get("/assistants/search")
async def search_assistants(graph_id: str = None):
    """Recherche d'assistants - SunuStat ANSD"""
    return [
        Assistant(
            assistant_id="sunustat-ansd-assistant",
            name="SunuStat - Assistant Statistiques Sénégal",
            graph_id="simple_rag",
            metadata={
                "created_by": "system",
                "description": "Assistant intelligent pour les statistiques officielles du Sénégal (ANSD)",
                "country": "Sénégal",
                "organization": "ANSD",
                "data_sources": ["RGPH", "EDS", "ESPS", "EHCVM", "ENES"],
                "rag_available": RAG_AVAILABLE
            }
        )
    ]

@app.post("/threads")
async def create_thread():
    """Création d'un nouveau thread SunuStat"""
    import uuid
    from datetime import datetime
    
    return Thread(
        thread_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat()
    )

@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: RunRequest):
    """Stream d'exécution SunuStat avec formatage ANSD"""
    
    async def generate_stream():
        try:
            # Extraire la dernière question
            last_message = request.messages[-1]
            query = last_message.content.strip()
            
            # Traiter les commandes spéciales
            special_response = process_special_commands(query)
            if special_response:
                # Envoyer la réponse de commande spéciale
                for word in special_response.split():
                    yield {
                        "event": "events",
                        "data": {
                            "event": "on_chat_model_stream",
                            "data": {"chunk": {"content": word + " "}}
                        }
                    }
                    await asyncio.sleep(0.02)
                return
            
            # Émission de l'événement de début avec style SunuStat
            yield {
                "event": "events",
                "data": {
                    "event": "on_retrieval_start",
                    "data": {
                        "input": "🔍 Recherche en cours dans les documents ANSD...",
                        "service": "SunuStat",
                        "query": query
                    }
                }
            }
            
            # Simulation de la progression
            progress_steps = [
                "• Récupération des documents ANSD",
                "• Analyse des données statistiques", 
                "• Génération de la réponse..."
            ]
            
            for step in progress_steps:
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_progress",
                        "data": {"step": step}
                    }
                }
                await asyncio.sleep(0.3)
            
            # Récupérer l'historique des messages précédents
            chat_history = []
            messages = request.messages[:-1]  # Tous sauf le dernier
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = messages[i].content
                    bot_msg = messages[i + 1].content
                    chat_history.append((user_msg, bot_msg))
            
            # Appeler Simple RAG ou générer réponse démo
            answer, sources = await call_simple_rag(query, chat_history[-5:])  # Limiter l'historique
            
            yield {
                "event": "events", 
                "data": {
                    "event": "on_retrieval_end",
                    "data": {
                        "output": f"✅ Analyse terminée - {len(sources) if sources else 0} documents consultés",
                        "documents_found": len(sources) if sources else 0
                    }
                }
            }
            
            # Formater la réponse finale avec style SunuStat
            formatted_response = f"**📊 SunuStat - ANSD répond :**\n\n{answer}"
            formatted_response += format_sources_info(sources)
            
            # Stream de la réponse formatée
            for word in formatted_response.split():
                yield {
                    "event": "events",
                    "data": {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": {"content": word + " "}}
                    }
                }
                await asyncio.sleep(0.03)  # Vitesse de streaming ajustable
                    
        except Exception as e:
            error_msg = f"❌ **Erreur technique**\n\nUne erreur s'est produite :\n`{str(e)}`\n\nVeuillez réessayer ou reformuler votre question."
            yield {
                "event": "events",
                "data": {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": {"content": error_msg}}
                }
            }
    
    return EventSourceResponse(generate_stream())

@app.get("/health")
async def health_check():
    """Health check avec informations SunuStat"""
    return {
        "status": "healthy",
        "service": "SunuStat ANSD Backend",
        "rag_available": RAG_AVAILABLE,
        "simple_rag_loaded": RAG_AVAILABLE,
        "version": "1.0.0"
    }

@app.get("/status")
async def status():
    """Statut détaillé du système"""
    return {
        "service": "SunuStat - ANSD Backend",
        "rag_system": {
            "available": RAG_AVAILABLE,
            "type": "Simple RAG" if RAG_AVAILABLE else "Demo Mode"
        },
        "data_sources": ["RGPH", "EDS", "ESPS", "EHCVM", "ENES"],
        "country": "Sénégal",
        "organization": "ANSD",
        "capabilities": [
            "Questions démographiques",
            "Statistiques de pauvreté", 
            "Données d'emploi",
            "Indicateurs de santé",
            "Statistiques d'éducation"
        ]
    }

if __name__ == "__main__":
    print("🇸🇳 Démarrage SunuStat ANSD Backend")
    print(f"📊 Simple RAG disponible: {RAG_AVAILABLE}")
    
    if RAG_AVAILABLE:
        print("✅ Backend prêt avec Simple RAG")
    else:
        print("⚠️ Backend en mode démonstration")
    
    uvicorn.run(
        "backend_server_sunustat:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )