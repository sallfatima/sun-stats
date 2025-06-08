# streamlit_app.py

import streamlit as st
from src.simple_rag.graph import RAGGraph

# --- Initialisation du chatbot ---
@st.cache_resource(show_spinner=False)
def init_chatbot():
    """
    Crée une instance unique du RAGGraph (chargement FAISS + LLM).
    """
    return RAGGraph()

chatbot = init_chatbot()

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Chatbot RGPH – ANSD",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Chatbot RGPH – ANSD (FAISS local)")
st.markdown(
    "Posez des questions sur les rapports RGPH et obtenez des réponses sourcées par FAISS + GPT-3.5."
)

# --- Gestion de l’historique de la conversation dans la session ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # liste de tuples (question, réponse)

# Champ de saisie utilisateur
question = st.text_input("Votre question :", key="input_question")

# Lorsque l’utilisateur appuie sur Entrée ou le bouton « Envoyer »
if st.button("Envoyer") and question.strip():
    with st.spinner("🤔 Recherche et génération en cours…"):
        # Récupère la réponse et les documents sources
        answer, sources = chatbot.ask(question, st.session_state.chat_history)
        # Enregistrer dans l’historique
        st.session_state.chat_history.append((question, answer, sources))
   

# Affichage de l’historique (dernières questions/réponses)
if st.session_state.chat_history:
    for i, (q, a, sources) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Vous :** {q}")
        st.markdown(f"**Bot :** {a}")
        # Afficher la liste des sources sous forme de liste à puces ± page/ID
        with st.expander("Voir les documents sources"):
            for doc in sources:
                src_pdf = doc.metadata.get("source_pdf", "inconnu")
                chunk_id = doc.metadata.get("chunk", "?")
                st.write(f"• `{src_pdf}` – chunk `{chunk_id}`")
        st.markdown("---")
