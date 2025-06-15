# =============================================================================
# FICHIER: chainlit_app.py - Version Complète avec Support Visuel
# =============================================================================

import chainlit as cl
import sys
import os
import base64
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio

# Ajouter le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# =============================================================================
# IMPORTS SÉCURISÉS ET CONFIGURATION
# =============================================================================

def verify_environment():
    """Vérifie que l'environnement est correctement configuré"""
    
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Variables d'environnement manquantes: {missing_vars}")
        return False
    
    print("✅ Variables d'environnement configurées")
    return True


def safe_import_simple_rag():
    """Import sécurisé du module simple_rag"""
    try:
        from simple_rag.graph import graph as simple_rag_graph
        from simple_rag.configuration import RagConfiguration
        print("✅ simple_rag importé avec succès")
        return simple_rag_graph, RagConfiguration, True
    except ImportError as e:
        print(f"❌ Erreur import simple_rag: {e}")
        return None, None, False


# Vérification et imports
if not verify_environment():
    print("⚠️ Veuillez configurer les variables d'environnement requises")
    sys.exit(1)

# Import du système RAG
simple_rag_graph, RagConfiguration, import_success = safe_import_simple_rag()

if not import_success:
    print("❌ Impossible d'importer simple_rag. Vérifiez votre installation.")
    sys.exit(1)

# =============================================================================
# FONCTIONS D'AFFICHAGE VISUEL
# =============================================================================

async def detect_and_display_visual_content(documents: List[Any], user_question: str) -> Tuple[List[Any], bool]:
    """
    Détecte et affiche automatiquement le contenu visuel des documents récupérés.
    
    Args:
        documents: Documents récupérés par le RAG
        user_question: Question de l'utilisateur
        
    Returns:
        Tuple (documents_textuels, contient_elements_visuels)
    """
    text_documents = []
    visual_elements = []
    
    # Parcourir tous les documents pour identifier les éléments visuels
    for doc in documents:
        metadata = getattr(doc, 'metadata', {})
        doc_type = metadata.get('type', '')
        source_type = metadata.get('source_type', '')
        
        # Identifier les documents visuels
        if (doc_type in ['visual_chart', 'visual_table'] or 
            source_type == 'visual' or
            'image_path' in metadata or 
            'table_path' in metadata):
            
            visual_element = {
                'document': doc,
                'type': doc_type,
                'metadata': metadata,
                'content': getattr(doc, 'page_content', '')
            }
            visual_elements.append(visual_element)
        else:
            text_documents.append(doc)
    
    # Afficher les éléments visuels trouvés
    if visual_elements:
        print(f"🎨 {len(visual_elements)} éléments visuels détectés pour: {user_question}")
        await display_visual_elements(visual_elements, user_question)
        return text_documents, True
    
    return text_documents, False


async def display_visual_elements(visual_elements: List[Dict], user_question: str) -> None:
    """
    Affiche les éléments visuels dans l'interface Chainlit.
    
    Args:
        visual_elements: Liste des éléments visuels à afficher
        user_question: Question de l'utilisateur (pour contexte)
    """
    # Message d'introduction pour les éléments visuels
    intro_msg = f"📊 **Éléments visuels ANSD trouvés :**\n*{user_question}*\n"
    
    for i, element in enumerate(visual_elements, 1):
        element_type = element['type']
        metadata = element['metadata']
        
        # Extraire les informations importantes
        caption = metadata.get('caption', f'Élément visuel {i}')
        pdf_name = metadata.get('pdf_name', 'Document ANSD')
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
        
        # Afficher selon le type
        try:
            if element_type == 'visual_chart':
                await display_chart_element(element, title, source_info, i)
            elif element_type == 'visual_table':
                await display_table_element(element, title, source_info, i)
            else:
                # Élément visuel générique
                await cl.Message(
                    content=f"{title}\n{source_info}\n\n📝 **Contenu :**\n{element['content'][:500]}..."
                ).send()
        except Exception as e:
            print(f"❌ Erreur affichage élément {i}: {e}")
            # Affichage de fallback
            await cl.Message(
                content=f"{title}\n{source_info}\n\n⚠️ *Erreur d'affichage de l'élément visuel*\n\n📝 **Contenu textuel :**\n{element['content'][:300]}..."
            ).send()


async def display_chart_element(element: Dict, title: str, source_info: str, index: int) -> None:
    """
    Affiche un graphique dans Chainlit.
    
    Args:
        element: Élément visuel de type graphique
        title: Titre formaté du graphique
        source_info: Informations sur la source
        index: Index de l'élément
    """
    metadata = element['metadata']
    image_path = metadata.get('image_path', '')
    
    if not image_path:
        # Pas de chemin d'image, afficher seulement le contenu textuel
        await cl.Message(
            content=f"{title}\n{source_info}\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()
        return
    
    # Vérifier que le fichier image existe
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"⚠️ Image non trouvée : {image_path}")
        await cl.Message(
            content=f"{title}\n{source_info}\n\n⚠️ *Image non disponible : {image_path}*\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()
        return
    
    try:
        # Lire et encoder l'image
        with open(image_file, 'rb') as f:
            image_data = f.read()
        
        # Créer l'élément image pour Chainlit
        image_element = cl.Image(
            name=f"chart_{index}",
            content=image_data,
            display="inline",
            size="large"
        )
        
        # Préparer le contenu du message
        content_parts = [title, source_info]
        
        if element['content']:
            # Limiter le contenu extrait pour l'affichage
            extracted_preview = element['content'][:400]
            if len(element['content']) > 400:
                extracted_preview += "..."
            content_parts.append(f"\n📝 **Contenu extrait :**\n{extracted_preview}")
        
        content = "\n".join(content_parts)
        
        # Afficher avec le message
        await cl.Message(
            content=content,
            elements=[image_element]
        ).send()
        
        print(f"✅ Graphique affiché : {image_file.name}")
        
    except Exception as e:
        print(f"❌ Erreur affichage graphique {image_path}: {e}")
        await cl.Message(
            content=f"{title}\n{source_info}\n\n❌ *Erreur lors du chargement de l'image*\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()


async def display_table_element(element: Dict, title: str, source_info: str, index: int) -> None:
    """
    Affiche un tableau dans Chainlit.
    
    Args:
        element: Élément visuel de type tableau
        title: Titre formaté du tableau
        source_info: Informations sur la source
        index: Index de l'élément
    """
    metadata = element['metadata']
    table_path = metadata.get('table_path', '')
    
    if not table_path:
        # Pas de chemin de tableau, afficher seulement le contenu textuel
        await cl.Message(
            content=f"{title}\n{source_info}\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()
        return
    
    # Vérifier que le fichier CSV existe
    table_file = Path(table_path)
    if not table_file.exists():
        print(f"⚠️ Tableau non trouvé : {table_path}")
        await cl.Message(
            content=f"{title}\n{source_info}\n\n⚠️ *Tableau non disponible : {table_path}*\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()
        return
    
    try:
        # Lire le CSV
        df = pd.read_csv(table_file)
        
        # Limiter l'affichage si le tableau est trop grand
        max_rows = 15
        max_cols = 8
        
        display_df = df.iloc[:max_rows, :max_cols]
        total_rows, total_cols = df.shape
        
        # Convertir en HTML pour l'affichage
        table_html = create_table_html(display_df, total_rows, total_cols, max_rows, max_cols)
        
        # Préparer le contenu
        content_parts = [title, source_info, "\n", table_html]
        
        if element['content']:
            content_parts.append(f"\n📝 **Données extraites :**\n{element['content'][:200]}...")
        
        content = "\n".join(content_parts)
        
        # Afficher le tableau
        await cl.Message(content=content).send()
        
        print(f"✅ Tableau affiché : {table_file.name} ({total_rows}x{total_cols})")
        
    except Exception as e:
        print(f"❌ Erreur affichage tableau {table_path}: {e}")
        await cl.Message(
            content=f"{title}\n{source_info}\n\n❌ *Erreur lors du chargement du tableau*\n\n📝 **Contenu extrait :**\n{element['content']}"
        ).send()


def create_table_html(df: pd.DataFrame, total_rows: int, total_cols: int, max_rows: int, max_cols: int) -> str:
    """
    Crée une représentation HTML d'un tableau pandas pour l'affichage dans Chainlit.
    """
    # Style CSS pour le tableau
    table_style = """
<style>
.ansd-table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 13px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}
.ansd-table th, .ansd-table td {
    border: 1px solid #e1e5e9;
    padding: 12px 8px;
    text-align: left;
    vertical-align: top;
}
.ansd-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
}
.ansd-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.ansd-table tr:hover {
    background-color: #e3f2fd;
    transition: background-color 0.2s ease;
}
.table-info {
    font-style: italic;
    color: #6c757d;
    margin-top: 10px;
    padding: 8px 12px;
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    border-radius: 0 4px 4px 0;
}
.ansd-table td {
    border-right: 1px solid #dee2e6;
}
.ansd-table td:last-child {
    border-right: none;
}
</style>
"""
    
    # Créer le tableau HTML
    html_parts = [table_style, '<table class="ansd-table">']
    
    # En-têtes
    html_parts.append('<thead><tr>')
    for col in df.columns:
        html_parts.append(f'<th>{str(col)}</th>')
    if total_cols > max_cols:
        html_parts.append('<th>...</th>')
    html_parts.append('</tr></thead>')
    
    # Corps du tableau
    html_parts.append('<tbody>')
    for _, row in df.iterrows():
        html_parts.append('<tr>')
        for value in row:
            # Nettoyer et formater la valeur
            if pd.isna(value):
                clean_value = ''
            else:
                clean_value = str(value)
                # Formater les nombres si nécessaire
                try:
                    if '.' in clean_value and clean_value.replace('.', '').replace('-', '').isdigit():
                        num_value = float(clean_value)
                        if num_value.is_integer():
                            clean_value = str(int(num_value))
                        else:
                            clean_value = f"{num_value:.2f}"
                except:
                    pass
            
            html_parts.append(f'<td>{clean_value}</td>')
        
        if total_cols > max_cols:
            html_parts.append('<td>...</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    html_parts.append('</table>')
    
    # Informations sur la troncature
    info_parts = []
    if total_rows > max_rows:
        info_parts.append(f"Affichage des {max_rows} premières lignes sur {total_rows}")
    if total_cols > max_cols:
        info_parts.append(f"Affichage des {max_cols} premières colonnes sur {total_cols}")
    
    if info_parts:
        html_parts.append(f'<div class="table-info">📊 {" • ".join(info_parts)}</div>')
    else:
        html_parts.append(f'<div class="table-info">📊 Tableau complet : {total_rows} lignes × {total_cols} colonnes</div>')
    
    return ''.join(html_parts)


# =============================================================================
# FONCTIONS PRINCIPALES DU CHATBOT
# =============================================================================

async def call_simple_rag(user_input: str, chat_history: list) -> Tuple[str, List[Any]]:
    """
    Appelle directement le module simple_rag avec support visuel.
    
    Args:
        user_input: Question de l'utilisateur
        chat_history: Historique des conversations
        
    Returns:
        Tuple (réponse, documents_récupérés)
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Convertir l'historique en messages LangChain
        messages = []
        for user_msg, bot_msg in chat_history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        
        # Ajouter le message actuel
        messages.append(HumanMessage(content=user_input))
        
        # Configuration pour simple_rag
        config = {
            "configurable": {
                "model": "openai/gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
                "pinecone_index": "ansd-doc",
                "retriever_provider": "pinecone",
                
                # Paramètres de récupération optimisés
                "retrieval_k": 20,
                "retrieval_fetch_k": 60,
                "enable_query_preprocessing": True,
                "enable_document_scoring": True,
                "prioritize_recent_data": True,
                
                # Support visuel
                "enable_visual_indexing": True,
                "images_dir": "images",
                "tables_dir": "tables"
            }
        }
        
        # Appeler simple_rag
        print(f"🤖 Appel simple_rag pour: {user_input}")
        result = await simple_rag_graph.ainvoke(
            {"messages": messages},
            config=config
        )
        
        # Extraire la réponse et les documents
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            answer = last_message.content
            
            # Vérifier les métadonnées visuelles
            response_metadata = getattr(last_message, 'response_metadata', {})
            has_visual = response_metadata.get('has_visual_content', False)
            
            print(f"✅ Réponse générée. Contenu visuel: {'Oui' if has_visual else 'Non'}")
        else:
            answer = "❌ Aucune réponse générée par le système."
        
        # Récupérer les documents pour l'affichage visuel
        documents = result.get("documents", [])
        
        return answer, documents
        
    except Exception as e:
        print(f"❌ Erreur dans call_simple_rag: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Erreur technique: {str(e)}", []


# =============================================================================
# INTERFACE CHAINLIT
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialisation du chat avec simple_rag"""
    
    await cl.Message(
        content="""🇸🇳 **Bienvenue dans TERANGA IA - ANSD**

**Assistant Intelligent pour les Statistiques du Sénégal**

✅ **Système simple_rag activé** avec support visuel automatique

📊 **Fonctionnalités :**
• Réponses expertes basées sur les données officielles ANSD
• Affichage automatique des graphiques et tableaux
• Citations précises des sources et pages
• Support des enquêtes : RGPH, EDS, ESPS, EHCVM, ENES

**💡 Exemples de questions :**
• *"Répartition des ménages par région selon la nature du revêtement du toit"*
• *"Évolution de la population du Sénégal par année"*
• *"Taux de pauvreté par région administrative"*
• *"Structure par âge de la population sénégalaise"*

**🎯 Posez votre question sur les statistiques du Sénégal !**"""
    ).send()
    
    # Initialiser l'historique
    cl.user_session.set("chat_history", [])
    
    print("✅ Session Chainlit initialisée avec simple_rag")


@cl.on_message
async def main(message):
    """Traitement principal des messages avec support visuel automatique"""
    
    try:
        # Récupérer l'historique
        chat_history = cl.user_session.get("chat_history", [])
        
        # Extraire le texte du message
        user_input = message.content
        
        print(f"📝 Question utilisateur: {user_input}")
        
        # Limiter l'historique envoyé (garder les 10 derniers échanges)
        short_history = chat_history[-10:]
        
        # Afficher un indicateur de traitement
        processing_msg = await cl.Message(
            content="🔍 **Recherche dans la base de données ANSD...**\n\n*Analyse en cours avec simple_rag et détection automatique du contenu visuel*"
        ).send()
        
        # Appeler simple_rag directement
        answer, documents = await call_simple_rag(user_input, short_history)
        
        # Supprimer le message de traitement
        await processing_msg.remove()
        
        # Détecter et afficher automatiquement le contenu visuel
        if documents:
            print(f"📄 {len(documents)} documents récupérés")
            text_docs, has_visual = await detect_and_display_visual_content(documents, user_input)
            
            if has_visual:
                # Ajouter une note à la réponse
                answer += "\n\n*📊 Les éléments visuels correspondants sont affichés ci-dessus.*"
        
        # Afficher la réponse textuelle
        await cl.Message(content=answer).send()
        
        # Mettre à jour l'historique
        chat_history.append((user_input, answer))
        
        # Limiter la taille de l'historique
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        cl.user_session.set("chat_history", chat_history)
        
        print(f"✅ Réponse envoyée. Historique: {len(chat_history)} échanges")
        
    except Exception as e:
        print(f"❌ Erreur dans main: {e}")
        import traceback
        traceback.print_exc()
        
        await cl.Message(
            content=f"❌ **Erreur technique**\n\nUne erreur s'est produite lors du traitement de votre question.\n\n*Détails: {str(e)}*\n\nVeuillez réessayer ou reformuler votre question."
        ).send()


# =============================================================================
# COMMANDES UTILITAIRES
# =============================================================================

@cl.on_message
async def handle_commands(message):
    """Gestion des commandes spéciales"""
    
    content = message.content.lower().strip()
    
    # Commande d'aide
    if content in ["/help", "/aide", "aide", "help"]:
        help_text = """🆘 **Aide - Assistant ANSD**

**📊 Types de questions supportées :**
• Statistiques démographiques (population, ménages, etc.)
• Données économiques (PIB, emploi, pauvreté, etc.)
• Indicateurs sociaux (éducation, santé, etc.)
• Répartitions géographiques (par région, milieu, etc.)
• Évolutions temporelles et tendances

**🎨 Affichage automatique :**
• Graphiques : Affichés automatiquement quand pertinents
• Tableaux : Formatés en HTML avec données complètes
• Sources : Citations précises avec PDF et page

**💡 Conseils pour de meilleures réponses :**
• Soyez spécifique : "population urbaine Dakar 2023"
• Mentionnez le type de données : "taux de pauvreté", "répartition"
• Précisez la zone : "par région", "milieu rural/urbain"

**🔧 Commandes disponibles :**
• `/help` ou `/aide` : Afficher cette aide
• `/debug` : Informations techniques
• `/clear` : Effacer l'historique

**📞 Support :** En cas de problème, reformulez votre question ou utilisez `/debug`"""
        
        await cl.Message(content=help_text).send()
        return
    
    # Commande de debug
    if content == "/debug":
        debug_info = f"""🔧 **Informations de Debug**

**🏗️ Configuration :**
• Système RAG : simple_rag activé
• Support visuel : ✅ Activé
• Modèle : gpt-4o-mini
• Index Pinecone : ansd-doc

**📁 Dossiers :**
• Images : {Path('images').exists() and len(list(Path('images').glob('*.png')))} fichiers
• Tableaux : {Path('tables').exists() and len(list(Path('tables').glob('*.csv')))} fichiers

**🔑 API :**
• OpenAI : {'✅ Configurée' if os.getenv('OPENAI_API_KEY') else '❌ Manquante'}
• Pinecone : {'✅ Configurée' if os.getenv('PINECONE_API_KEY') else '❌ Manquante'}

**💾 Session :**
• Historique : {len(cl.user_session.get('chat_history', []))} échanges"""
        
        await cl.Message(content=debug_info).send()
        return
    
    # Commande de nettoyage
    if content == "/clear":
        cl.user_session.set("chat_history", [])
        await cl.Message(content="🧹 **Historique effacé**\n\nL'historique de conversation a été remis à zéro.").send()
        return


# =============================================================================
# DÉMARRAGE DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Lancement de TERANGA IA - ANSD")
    print("📊 Système simple_rag avec support visuel")
    print("🔗 Interface Chainlit prête")
    
    # L'application sera lancée avec: chainlit run chainlit_app.py