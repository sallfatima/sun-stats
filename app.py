# streamlit_app.py

import streamlit as st
import sys
import os
import asyncio

# Ajouter le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import direct du Simple RAG
try:
    from simple_rag.graph import graph as simple_rag_graph
    from langchain_core.messages import HumanMessage, AIMessage
    RAG_AVAILABLE = True
    print("✅ Simple RAG chargé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import Simple RAG: {e}")
    RAG_AVAILABLE = False

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="SunuStat - ANSD",
    page_icon="🇸🇳",
    layout="wide",
)

# Fonction pour appeler Simple RAG
async def call_simple_rag(user_input: str, chat_history: list):
    """Appelle le Simple RAG avec le message utilisateur"""
    
    if not RAG_AVAILABLE:
        return "❌ Simple RAG non disponible", []
    
    try:
        # Convertir l'historique en messages LangChain
        messages = []
        for user_msg, bot_msg, _ in chat_history:  # Ignorer les sources dans l'historique
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        # Configuration par défaut
        config = None
        
        print(f"🔍 Appel Simple RAG avec {len(messages)} messages")
        
        # Appeler le graphique Simple RAG
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

# Fonction wrapper synchrone pour Streamlit
def ask_simple_rag(user_input: str, chat_history: list):
    """Wrapper synchrone pour appeler Simple RAG depuis Streamlit"""
    try:
        # Créer un nouvel event loop s'il n'existe pas
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Exécuter la fonction async
        return loop.run_until_complete(call_simple_rag(user_input, chat_history))
    except Exception as e:
        print(f"❌ Erreur wrapper: {e}")
        return f"❌ Erreur: {str(e)}", []

# --- Interface principale ---
st.title("🇸🇳 SunuStat - ANSD")

# Message de bienvenue
st.markdown("""
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
""")

# Vérification de la disponibilité de Simple RAG
if not RAG_AVAILABLE:
    st.error("❌ **Simple RAG non disponible**\n\nVérifiez que le module simple_rag est correctement installé.")
    st.stop()

# --- Gestion de l'historique de la conversation dans la session ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # liste de tuples (question, réponse, sources)

# Sidebar pour l'aide et les commandes
with st.sidebar:
    st.header("🆘 Aide")
    st.markdown("""
    **Types de données disponibles :**
    • **Démographiques** - Population, natalité, mortalité
    • **Économiques** - PIB, pauvreté, emploi, croissance
    • **Sociales** - Éducation, santé, alphabétisation
    • **Géographiques** - Régions, départements, communes
    
    **Types d'enquêtes ANSD :**
    • **RGPH** - Recensement (données population/habitat)
    • **EDS** - Enquête Démographique et Santé
    • **ESPS** - Enquête Suivi Pauvreté Sénégal
    • **EHCVM** - Enquête Conditions Vie Ménages
    • **ENES** - Enquête Nationale Emploi Sénégal
    """)
    
    if st.button("🧹 Effacer l'historique", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# --- Interface de chat ---
st.subheader("💬 Posez votre question")

# Champ de saisie utilisateur
with st.form("question_form", clear_on_submit=True):
    question = st.text_area(
        "Votre question sur les statistiques du Sénégal :", 
        placeholder="Ex: Quelle est la population du Sénégal selon le dernier RGPH ?",
        height=100
    )
    submit_button = st.form_submit_button("📤 Envoyer", type="primary")

# Traitement de la question
if submit_button and question.strip():
    with st.spinner("🔍 Recherche dans la base de données ANSD..."):
        try:
            # Récupère la réponse et les documents sources
            answer, sources = ask_simple_rag(question, st.session_state.chat_history)
            
            # Enregistrer dans l'historique
            st.session_state.chat_history.append((question, answer, sources))
            
            # Recharger la page pour afficher la nouvelle réponse
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement: {str(e)}")

# --- Affichage de l'historique ---
if st.session_state.chat_history:
    st.subheader("📜 Historique de la conversation")
    
    # Afficher les conversations de la plus récente à la plus ancienne
    for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history)):
        conversation_number = len(st.session_state.chat_history) - i
        
        with st.container():
            # Question de l'utilisateur
            st.markdown(f"**🧑‍💼 Question {conversation_number} :** {q}")
            
            # Réponse du bot
            st.markdown(f"**🤖 SunuStat répond :**")
            st.markdown(a)
            
            # Sources avec informations détaillées
            if sources and len(sources) > 0:
                with st.expander(f"📚 Voir les sources consultées ({len(sources)} document(s))"):
                    st.markdown("**Documents ANSD utilisés pour cette réponse :**")
                    
                    for idx, doc in enumerate(sources, 1):
                        if hasattr(doc, 'metadata') and doc.metadata:
                            # Extraire les métadonnées
                            pdf_name = doc.metadata.get('pdf_name', 'Document ANSD')
                            page_num = doc.metadata.get('page_num', 'N/A')
                            source = doc.metadata.get('source', 'N/A')
                            
                            # Nettoyer le nom du PDF
                            if '/' in pdf_name:
                                pdf_name = pdf_name.split('/')[-1]
                            
                            # Afficher les informations de source
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"📄 **{idx}.** {pdf_name}")
                            with col2:
                                if page_num != 'N/A':
                                    st.write(f"📖 Page {page_num}")
                                else:
                                    st.write("📖 Page N/A")
                            with col3:
                                st.write("🏛️ ANSD")
                            
                            # Aperçu du contenu (optionnel)
                            if hasattr(doc, 'page_content') and doc.page_content:
                                if st.button(f"👁️ Voir le contenu - Source {idx}", key=f"content_{conversation_number}_{idx}"):
                                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.text_area(f"Contenu de la source {idx}:", content_preview, height=100, disabled=True)
                        else:
                            st.write(f"📄 **{idx}.** Document ANSD (métadonnées non disponibles)")
            else:
                st.info("📝 Cette réponse est basée sur les connaissances générales ANSD (aucun document spécifique récupéré)")
            
            st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🇸🇳 <strong>SunuStat - ANSD</strong> | Assistant Intelligent pour les Statistiques du Sénégal</p>
    <p>Propulsé par Simple RAG | Données officielles ANSD</p>
</div>
""", unsafe_allow_html=True)