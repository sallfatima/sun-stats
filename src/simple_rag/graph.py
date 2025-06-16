# =============================================================================
# src/simple_rag/graph.py - VERSION COMPLÈTE AVEC VISUELS ET SUGGESTIONS
# =============================================================================

### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

# Imports pour l'affichage visuel (conditionnel selon votre framework)
try:
    # Pour Chainlit
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False

try:
    # Pour Streamlit
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# =============================================================================
# PROMPT SYSTEM AMÉLIORÉ POUR L'ANSD
# =============================================================================

IMPROVED_ANSD_SYSTEM_PROMPT = """Vous êtes un expert statisticien de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal), spécialisé dans l'analyse de données démographiques, économiques et sociales du Sénégal.

MISSION PRINCIPALE :
Répondre de manière complète et approfondie aux questions sur les statistiques du Sénégal en utilisant PRIORITAIREMENT les documents fournis et en complétant avec vos connaissances des publications officielles de l'ANSD.

SOURCES AUTORISÉES :
✅ Documents fournis dans le contexte (PRIORITÉ ABSOLUE)
✅ Connaissances des rapports officiels ANSD publiés
✅ Données du site officiel ANSD (www.ansd.sn)
✅ Publications officielles des enquêtes ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Comptes nationaux et statistiques économiques officielles du Sénégal
✅ Projections démographiques officielles de l'ANSD

❌ SOURCES INTERDITES :
❌ Données d'autres pays pour combler les lacunes
❌ Estimations personnelles non basées sur les sources ANSD
❌ Informations non officielles ou de sources tierces
❌ Projections personnelles non documentées

RÈGLES DE RÉDACTION :
✅ Réponse directe : SANS limitation de phrases - développez autant que nécessaire
✅ Contexte additionnel : SANS limitation - incluez toutes les informations pertinentes
✅ Citez TOUJOURS vos sources précises (document + page ou publication ANSD)
✅ Distinguez clairement les données des documents fournis vs connaissances ANSD
✅ Donnez les chiffres EXACTS quand disponibles
✅ Précisez SYSTÉMATIQUEMENT les années de référence
✅ Mentionnez les méthodologies d'enquête

FORMAT DE RÉPONSE OBLIGATOIRE :

**RÉPONSE DIRECTE :**
[Développez la réponse de manière complète et détaillée, sans limitation de longueur. Incluez tous les éléments pertinents pour une compréhension approfondie du sujet. Vous pouvez utiliser plusieurs paragraphes et développer les aspects importants.]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeur exacte avec unité]
- Année de référence : [année précise]
- Source : [nom exact du document, page X OU publication ANSD officielle]
- Méthodologie : [enquête/recensement utilisé]

**CONTEXTE ADDITIONNEL :**
[Développez largement avec toutes les informations complémentaires pertinentes, sans limitation de longueur.]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation, changements méthodologiques, définitions spécifiques]

DOCUMENTS ANSD DISPONIBLES :
{context}

Analysez maintenant ces documents et répondez à la question de l'utilisateur de manière complète et approfondie."""

# =============================================================================
# FONCTIONS DE DÉTECTION ET TRAITEMENT DES ÉLÉMENTS VISUELS
# =============================================================================

def extract_visual_elements(documents):
    """Sépare les documents textuels et les éléments visuels - VERSION AMÉLIORÉE."""
    
    text_docs = []
    visual_elements = []
    
    print(f"🔍 Analyse de {len(documents)} documents pour éléments visuels...")
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # Diagnostic: afficher les métadonnées importantes pour debug
        if i <= 3:  # Pour les 3 premiers documents
            print(f"  📄 Document {i}: {metadata.get('pdf_name', 'Unknown')[:30]}")
            important_keys = ['type', 'image_path', 'chart_type', 'is_table', 'visual_type', 'source']
            for key in important_keys:
                if key in metadata:
                    print(f"    {key}: {metadata[key]}")
        
        # Détection améliorée des éléments visuels
        is_visual, element_type = detect_visual_element_enhanced(doc, metadata, content)
        
        if is_visual:
            visual_element = {
                'type': element_type,
                'metadata': metadata,
                'content': content,
                'document': doc,
                'relevance_score': 0  # Sera calculé plus tard
            }
            visual_elements.append(visual_element)
            
            if i <= 3:
                print(f"    🎨 VISUEL DÉTECTÉ: {element_type}")
        else:
            text_docs.append(doc)
            if i <= 3:
                print(f"    📝 TEXTE")
    
    print(f"✅ Résultat: {len(text_docs)} textuels, {len(visual_elements)} visuels")
    return text_docs, visual_elements

def detect_visual_element_enhanced(doc, metadata, content):
    """Détection améliorée des éléments visuels selon la structure de vos données."""
    
    # MÉTHODE 1: Métadonnées explicites de type visuel
    visual_type_indicators = {
        'image_path': 'visual_chart',
        'chart_type': 'visual_chart', 
        'visual_type': 'visual_chart',
        'is_table': 'visual_table',
        'table_data': 'visual_table',
        'chart_category': 'visual_chart',
        'source_type': 'visual_chart'  # Ajout pour compatibilité
    }
    
    for indicator, visual_type in visual_type_indicators.items():
        if indicator in metadata and metadata[indicator]:
            return True, visual_type
    
    # MÉTHODE 2: Type de document explicite
    doc_type = metadata.get('type', '').lower()
    visual_types = {
        'visual_chart': 'visual_chart',
        'visual_table': 'visual_table',
        'image': 'visual_chart',
        'chart': 'visual_chart',
        'table': 'visual_table',
        'graph': 'visual_chart',
        'figure': 'visual_chart'
    }
    
    if doc_type in visual_types:
        return True, visual_types[doc_type]
    
    # MÉTHODE 3: Analyse du nom de fichier source
    source = metadata.get('source', '').lower()
    pdf_name = metadata.get('pdf_name', '').lower()
    
    # Patterns pour images
    image_patterns = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']
    for pattern in image_patterns:
        if pattern in source:
            return True, 'visual_chart'
    
    # Patterns pour tableaux
    table_patterns = ['.csv', 'table', 'tableau']
    for pattern in table_patterns:
        if pattern in source or pattern in pdf_name:
            return True, 'visual_table'
    
    # Patterns pour graphiques
    chart_patterns = ['chart', 'graph', 'figure', 'diagramme', 'graphique']
    for pattern in chart_patterns:
        if pattern in source or pattern in pdf_name:
            return True, 'visual_chart'
    
    # MÉTHODE 4: Analyse du contenu textuel pour tableaux
    if content and is_table_content_enhanced(content):
        return True, 'visual_table'
    
    # MÉTHODE 5: Mots-clés visuels dans le contenu
    if content:
        content_lower = content.lower()
        
        # Indicateurs de graphiques
        chart_keywords = [
            'graphique', 'figure', 'diagramme', 'courbe', 'histogramme', 
            'secteur', 'barres', 'camembert', 'visualisation', 'chart'
        ]
        
        chart_score = sum(1 for keyword in chart_keywords if keyword in content_lower)
        if chart_score >= 2:  # Au moins 2 indicateurs
            return True, 'visual_chart'
        
        # Indicateurs de description d'image
        image_keywords = [
            'image extraite', 'page', 'png', 'jpg', 'capture', 'screenshot'
        ]
        
        if any(keyword in content_lower for keyword in image_keywords):
            return True, 'visual_chart'
    
    # MÉTHODE 6: Patterns dans les métadonnées pages spécifiques
    page_num = metadata.get('page_num', metadata.get('page', ''))
    if page_num and ('img' in str(page_num) or 'chart' in str(page_num)):
        return True, 'visual_chart'
    
    return False, None

def is_table_content_enhanced(content: str) -> bool:
    """Détection améliorée du contenu tabulaire."""
    
    if not content or len(content.strip()) < 50:
        return False
    
    lines = content.split('\n')
    if len(lines) < 3:  # Minimum 3 lignes pour un tableau
        return False
    
    # Compteurs d'indicateurs de tableau
    pipe_lines = sum(1 for line in lines if '|' in line)
    tab_lines = sum(1 for line in lines if '\t' in line)
    number_lines = sum(1 for line in lines if re.search(r'\d+', line))
    
    # Mots-clés de tableau
    table_keywords = [
        'total', 'sous-total', 'colonnes:', 'ligne ', 'données:',
        'tableau', 'pourcentage', '%', 'millions', 'milliards'
    ]
    
    keyword_score = sum(1 for keyword in table_keywords if keyword.lower() in content.lower())
    
    # Lignes avec structure de colonnes (espaces multiples)
    column_lines = sum(1 for line in lines if re.search(r'\s{3,}', line))
    
    # Score composite
    table_score = 0
    
    if pipe_lines >= 2:
        table_score += 3
    if tab_lines >= 2:
        table_score += 3
    if column_lines >= 3:
        table_score += 2
    if number_lines >= 3:
        table_score += 1
    if keyword_score >= 2:
        table_score += 2
    
    # Patterns de données ANSD
    ansd_patterns = [
        r'\d+\s*%',  # Pourcentages
        r'\d+\s*millions?',  # Millions
        r'\d+\s*milliards?',  # Milliards
        r'20\d{2}',  # Années
        r'région|département|urbain|rural'  # Géographie
    ]
    
    ansd_score = sum(1 for pattern in ansd_patterns if re.search(pattern, content, re.IGNORECASE))
    if ansd_score >= 2:
        table_score += 2
    
    return table_score >= 4

def analyze_visual_relevance_enhanced(visual_elements: List[Dict[str, Any]], user_question: str) -> List[Dict[str, Any]]:
    """Analyse améliorée de la pertinence des éléments visuels."""
    
    if not visual_elements:
        return []
    
    question_lower = user_question.lower()
    relevant_elements = []
    
    print(f"🎯 Analyse de pertinence pour {len(visual_elements)} éléments visuels...")
    
    # Mots-clés thématiques ANSD étendus
    theme_keywords = {
        'démographie': [
            'population', 'habitants', 'démographique', 'natalité', 'mortalité', 
            'âge', 'sexe', 'recensement', 'rgph'
        ],
        'économie': [
            'économie', 'pib', 'croissance', 'secteur', 'activité', 'revenus',
            'production', 'commerce', 'industrie'
        ],
        'emploi': [
            'emploi', 'travail', 'chômage', 'actifs', 'profession', 'occupation',
            'métier', 'activité', 'enes'
        ],
        'pauvreté': [
            'pauvreté', 'pauvre', 'indigence', 'vulnérabilité', 'revenus',
            'ménage', 'esps', 'conditions'
        ],
        'éducation': [
            'éducation', 'école', 'scolarisation', 'alphabétisation', 'instruction',
            'enseignement', 'formation'
        ],
        'santé': [
            'santé', 'mortalité', 'vaccination', 'maternelle', 'morbidité',
            'médical', 'eds', 'sanitaire'
        ],
        'géographie': [
            'région', 'département', 'urbain', 'rural', 'ville', 'commune',
            'territorial', 'localité', 'zone'
        ]
    }
    
    for i, element in enumerate(visual_elements, 1):
        relevance_score = 0
        metadata = element['metadata']
        content = element['content'].lower()
        element_type = element['type']
        
        print(f"  📊 Élément {i}: {metadata.get('pdf_name', 'Unknown')[:30]}")
        
        # Score 1: Correspondance thématique
        theme_score = 0
        for theme, keywords in theme_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                theme_matches = sum(1 for keyword in keywords if keyword in content)
                if theme_matches > 0:
                    theme_score += theme_matches
                    print(f"    🎯 Thème {theme}: {theme_matches} correspondances")
        
        relevance_score += min(theme_score, 5)  # Max 5 points pour thème
        
        # Score 2: Mots-clés directs de la question
        question_words = set(word for word in question_lower.split() if len(word) > 3)
        content_words = set(content.split())
        common_words = question_words.intersection(content_words)
        
        word_score = len(common_words)
        relevance_score += min(word_score, 3)  # Max 3 points pour mots
        
        if common_words:
            print(f"    🔤 Mots communs: {', '.join(list(common_words)[:3])}")
        
        # Score 3: Type d'élément vs type de question
        if element_type == 'visual_table' and any(word in question_lower for word in ['combien', 'nombre', 'taux', 'pourcentage', 'données']):
            relevance_score += 2
            print(f"    📋 Bonus tableau pour question quantitative")
        
        if element_type == 'visual_chart' and any(word in question_lower for word in ['évolution', 'tendance', 'graphique', 'comparaison']):
            relevance_score += 2
            print(f"    📈 Bonus graphique pour question visuelle")
        
        # Score 4: Métadonnées spécifiques
        pdf_name = metadata.get('pdf_name', '').lower()
        if any(word in pdf_name for word in question_words):
            relevance_score += 2
            print(f"    📄 Bonus nom de fichier")
        
        # Score 5: Année/période
        if re.search(r'20\d{2}', question_lower) and re.search(r'20\d{2}', content):
            question_years = set(re.findall(r'20\d{2}', question_lower))
            content_years = set(re.findall(r'20\d{2}', content))
            if question_years.intersection(content_years):
                relevance_score += 3
                print(f"    📅 Bonus année correspondante")
        
        print(f"    ⭐ Score total: {relevance_score}")
        
        # Seuil de pertinence adaptatif
        min_threshold = 3 if len(visual_elements) > 10 else 2
        
        if relevance_score >= min_threshold:
            element['relevance_score'] = relevance_score
            relevant_elements.append(element)
            print(f"    ✅ RETENU")
        else:
            print(f"    ❌ Rejeté (seuil: {min_threshold})")
    
    # Trier par pertinence et limiter
    relevant_elements.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    max_elements = 5  # Maximum 5 éléments visuels
    
    final_elements = relevant_elements[:max_elements]
    
    print(f"🎯 Résultat final: {len(final_elements)} éléments pertinents sélectionnés")
    
    return final_elements

# =============================================================================
# FONCTIONS D'AFFICHAGE VISUEL
# =============================================================================

async def display_visual_element(element: Dict[str, Any], user_question: str) -> bool:
    """Affiche un élément visuel (graphique ou tableau)."""
    
    element_type = element['type']
    metadata = element['metadata']
    content = element['content']
    
    try:
        if element_type == 'visual_chart':
            return await display_chart_element(metadata, content)
        elif element_type == 'visual_table':
            return await display_table_element(metadata, content)
        return False
    except Exception as e:
        print(f"❌ Erreur affichage élément visuel: {e}")
        return False

async def display_chart_element(metadata: Dict[str, Any], content: str) -> bool:
    """Affiche un graphique avec recherche intelligente du chemin de l'image."""
    
    # Informations du graphique
    pdf_name = metadata.get('pdf_name', 'Document ANSD')
    page_num = metadata.get('page_num', metadata.get('page', 'N/A'))
    chart_type = metadata.get('chart_type', 'graphique')
    
    # Titre et source
    title = f"📊 **{chart_type.title()}**"
    source_info = f"*Source: {pdf_name}"
    if page_num != 'N/A':
        source_info += f", page {page_num}"
    source_info += "*"
    
    print(f"🎨 Tentative d'affichage: {title}")
    print(f"🔍 Métadonnées image: {metadata}")
    
    # RECHERCHE INTELLIGENTE DU CHEMIN DE L'IMAGE
    image_path = None
    
    # Méthode 1: Chercher image_path dans les métadonnées
    for key in ['image_path', 'source', 'file_path', 'path']:
        if key in metadata and metadata[key]:
            potential_path = metadata[key]
            print(f"🔍 Test chemin {key}: {potential_path}")
             
            if Path(potential_path).exists():
                image_path = potential_path
                print(f"✅ Image trouvée: {image_path}")
                break
            else:
                # Essayer avec différents préfixes
                filename = Path(potential_path).name
                for base_dir in ['images/', 'data/images/', './images/', '']:
                    test_path = Path(base_dir) / filename
                    if test_path.exists():
                        image_path = str(test_path)
                        print(f"✅ Image trouvée (chemin alternatif): {image_path}")
                        break
                
                if image_path:
                    break
    
    # Méthode 2: Construire le chemin à partir du nom du PDF
    if not image_path and pdf_name and page_num != 'N/A':
        import glob
        
        # Patterns typiques pour les images extraites
        pdf_stem = Path(pdf_name).stem if '.' in pdf_name else pdf_name
        patterns = [
            f"images/{pdf_stem}*page*{page_num}*.png",
            f"images/{pdf_stem}*p{page_num}*.png", 
            f"data/images/{pdf_stem}*page*{page_num}*.png",
            f"images/*{pdf_stem}*{page_num}*.png",
            f"images/*page*{page_num}*.png"
        ]
        
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
                print(f"✅ Image trouvée par pattern: {image_path}")
                break
    
    # Méthode 3: Chercher par contenu (si le contenu mentionne un nom de fichier)
    if not image_path:
        import re
        filename_match = re.search(r'([a-zA-Z0-9_-]+\.(?:png|jpg|jpeg))', content)
        if filename_match:
            filename = filename_match.group(1)
            for base_dir in ['images/', 'data/images/', './images/']:
                test_path = Path(base_dir) / filename
                if test_path.exists():
                    image_path = str(test_path)
                    print(f"✅ Image trouvée par contenu: {image_path}")
                    break
    
    # AFFICHAGE DE L'IMAGE OU FALLBACK
    if image_path and Path(image_path).exists():
        try:
            if CHAINLIT_AVAILABLE:
                # Message avec titre et source
                await cl.Message(content=f"{title}\n\n{source_info}").send()
                
                # Afficher l'image
                elements = [cl.Image(name="chart", path=str(image_path), display="inline")]
                await cl.Message(content="", elements=elements).send()
                
                # Message de confirmation
                await cl.Message(content=f"✅ Graphique affiché depuis: `{image_path}`").send()
                
                print(f"✅ Image Chainlit affichée: {image_path}")
                return True
                
            elif STREAMLIT_AVAILABLE:
                st.markdown(f"### {title}")
                st.markdown(source_info)
                st.image(str(image_path), caption=f"Source: {image_path}")
                print(f"✅ Image Streamlit affichée: {image_path}")
                return True
                
            else:
                # Affichage console avec chemin de l'image
                print(f"\n{title}")
                print(source_info)
                print(f"🖼️ IMAGE DISPONIBLE: {image_path}")
                print(f"📝 Description: {content[:200]}...")
                return True
                
        except Exception as e:
            print(f"❌ Erreur affichage image: {e}")
    
    # FALLBACK: Afficher la description textuelle avec diagnostic
    print(f"⚠️ Image non trouvée, affichage de la description")
    
    # Message de diagnostic
    diagnostic_info = f"""
{title}

{source_info}

⚠️ **Image non accessible**
🔍 Chemins recherchés:
"""
    
    # Ajouter les chemins testés au diagnostic
    for key in ['image_path', 'source', 'file_path']:
        if key in metadata:
            diagnostic_info += f"   • {key}: `{metadata[key]}`\n"
    
    diagnostic_info += f"\n📝 **Description du graphique :**\n{content}"
    
    if CHAINLIT_AVAILABLE:
        await cl.Message(content=diagnostic_info).send()
    elif STREAMLIT_AVAILABLE:
        st.markdown(diagnostic_info)
    else:
        print(diagnostic_info)
    
    return True

# =============================================================================
# AJOUTEZ AUSSI CETTE FONCTION DE DEBUG DANS VOTRE GRAPH.PY
# =============================================================================

def debug_visual_elements(visual_elements):
    """Debug des éléments visuels pour diagnostic."""
    
    print(f"\n🔍 DEBUG: {len(visual_elements)} éléments visuels détectés")
    print("=" * 50)
    
    for i, element in enumerate(visual_elements, 1):
        metadata = element['metadata']
        print(f"\n📊 Élément {i}:")
        print(f"   Type: {element['type']}")
        print(f"   PDF: {metadata.get('pdf_name', 'N/A')}")
        print(f"   Page: {metadata.get('page_num', metadata.get('page', 'N/A'))}")
        print(f"   Image path: {metadata.get('image_path', 'N/A')}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
        print(f"   Contenu (aperçu): {element['content'][:100]}...")
        
        # Vérifier si l'image existe
        image_path = metadata.get('image_path')
        if image_path:
            if Path(image_path).exists():
                print(f"   ✅ Image accessible")
            else:
                print(f"   ❌ Image manquante")
                
                # Suggérer des chemins alternatifs
                filename = Path(image_path).name
                alternatives = [
                    f"images/{filename}",
                    f"data/images/{filename}",
                    f"./images/{filename}"
                ]
                
                for alt in alternatives:
                    if Path(alt).exists():
                        print(f"   💡 Trouvée à: {alt}")
                        break
async def display_chart_chainlit(metadata: Dict[str, Any], title: str, source_info: str, content: str) -> bool:
    """Affichage graphique pour Chainlit."""
    
    try:
        # Message avec titre et source
        await cl.Message(content=f"{title}\n\n{source_info}").send()
        
        # Afficher l'image si disponible
        image_path = metadata.get('image_path')
        if image_path and Path(image_path).exists():
            elements = [cl.Image(name="chart", path=str(image_path), display="inline")]
            await cl.Message(content="", elements=elements).send()
        else:
            # Fallback: afficher la description textuelle
            await cl.Message(content=f"📈 Description: {content[:500]}...").send()
        
        return True
    except Exception as e:
        print(f"❌ Erreur Chainlit chart: {e}")
        return False

def display_chart_streamlit(metadata: Dict[str, Any], title: str, source_info: str, content: str) -> bool:
    """Affichage graphique pour Streamlit."""
    
    try:
        st.markdown(f"### {title}")
        st.markdown(source_info)
        
        # Afficher l'image si disponible
        image_path = metadata.get('image_path')
        if image_path and Path(image_path).exists():
            st.image(str(image_path))
        else:
            # Fallback: afficher la description
            st.text(content[:500] + "..." if len(content) > 500 else content)
        
        return True
    except Exception as e:
        print(f"❌ Erreur Streamlit chart: {e}")
        return False

def display_chart_text(metadata: Dict[str, Any], title: str, source_info: str, content: str) -> bool:
    """Affichage textuel pour les graphiques."""
    
    print(f"\n{title}")
    print(source_info)
    print(f"📈 Contenu: {content[:300]}...")
    return True

async def display_table_element(metadata: Dict[str, Any], content: str) -> bool:
    """Affiche un tableau."""
    
    # Informations du tableau
    pdf_name = metadata.get('pdf_name', 'Document ANSD')
    page_num = metadata.get('page_num', metadata.get('page', 'N/A'))
    
    # Titre et source
    title = "📋 **Tableau de données**"
    source_info = f"*Source: {pdf_name}"
    if page_num != 'N/A':
        source_info += f", page {page_num}"
    source_info += "*"
    
    # Affichage selon le framework
    if CHAINLIT_AVAILABLE:
        return await display_table_chainlit(content, title, source_info)
    elif STREAMLIT_AVAILABLE:
        return display_table_streamlit(content, title, source_info)
    else:
        return display_table_text(content, title, source_info)

async def display_table_chainlit(content: str, title: str, source_info: str) -> bool:
    """Affichage tableau pour Chainlit."""
    
    try:
        # Titre et source
        await cl.Message(content=f"{title}\n\n{source_info}").send()
        
        # Formater le tableau pour l'affichage
        formatted_table = format_table_for_display(content)
        await cl.Message(content=f"```\n{formatted_table}\n```").send()
        
        return True
    except Exception as e:
        print(f"❌ Erreur Chainlit table: {e}")
        return False

def display_table_streamlit(content: str, title: str, source_info: str) -> bool:
    """Affichage tableau pour Streamlit."""
    
    try:
        st.markdown(f"### {title}")
        st.markdown(source_info)
        
        # Essayer de parser le tableau en DataFrame
        try:
            import pandas as pd
            # Logique simple pour convertir le contenu en DataFrame
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) > 1:
                # Première ligne = headers
                headers = re.split(r'\s{2,}|\t|\|', lines[0])
                data = []
                for line in lines[1:]:
                    row = re.split(r'\s{2,}|\t|\|', line)
                    if len(row) == len(headers):
                        data.append(row)
                
                if data:
                    df = pd.DataFrame(data, columns=headers)
                    st.dataframe(df)
                    return True
        except:
            pass
        
        # Fallback: affichage texte formaté
        formatted_table = format_table_for_display(content)
        st.text(formatted_table)
        return True
        
    except Exception as e:
        print(f"❌ Erreur Streamlit table: {e}")
        return False

def display_table_text(content: str, title: str, source_info: str) -> bool:
    """Affichage textuel pour les tableaux."""
    
    print(f"\n{title}")
    print(source_info)
    formatted_table = format_table_for_display(content)
    print(f"📋 Tableau:\n{formatted_table}")
    return True
def debug_visual_elements(visual_elements):
    """Debug des éléments visuels pour diagnostic."""
    
    print(f"\n🔍 DEBUG VISUAL: {len(visual_elements)} éléments détectés")
    print("=" * 50)
    
    for i, element in enumerate(visual_elements, 1):
        metadata = element['metadata']
        print(f"\n📊 Élément {i}:")
        print(f"   Type: {element['type']}")
        print(f"   PDF: {metadata.get('pdf_name', 'N/A')}")
        print(f"   Page: {metadata.get('page_num', metadata.get('page', 'N/A'))}")
        
        # Afficher TOUTES les métadonnées qui pourraient contenir un chemin
        image_keys = {}
        for key, value in metadata.items():
            if any(term in key.lower() for term in ['image', 'path', 'file', 'source']):
                image_keys[key] = value
        
        if image_keys:
            print(f"   🖼️ Métadonnées d'image:")
            for key, value in image_keys.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Aucune métadonnée d'image")
        
        print(f"   📝 Contenu: {element['content'][:100]}...")

def check_graph_corrections():
    """Vérifie si les corrections sont appliquées dans graph.py."""
    
    print("🔍 VÉRIFICATION DES CORRECTIONS DANS graph.py")
    print("=" * 50)
    
    try:
        with open('src/simple_rag/graph.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier la présence des corrections
        corrections = {
            'debug_visual_elements': 'def debug_visual_elements' in content,
            'find_image_path_smart': 'def find_image_path_smart' in content or 'glob.glob' in content,
            'enhanced_display': 'image_path = None' in content and 'Path(image_path).exists()' in content,
            'metadata_debug': 'Métadonnées complètes' in content or 'métadonnées d\'image' in content
        }
        
        print("Corrections appliquées:")
        for correction, present in corrections.items():
            status = "✅" if present else "❌"
            print(f"   {status} {correction}")
        
        if all(corrections.values()):
            print("\n✅ Toutes les corrections sont appliquées!")
            return True
        else:
            print("\n⚠️ Certaines corrections manquent")
            return False
            
    except FileNotFoundError:
        print("❌ Fichier graph.py non trouvé dans src/simple_rag/")
        return False
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        return False

async def display_chart_element(metadata: Dict[str, Any], content: str) -> bool:
    """Affiche un graphique avec recherche intelligente du chemin de l'image."""
    
    # Informations du graphique
    pdf_name = metadata.get('pdf_name', 'Document ANSD')
    page_num = metadata.get('page_num', metadata.get('page', 'N/A'))
    chart_type = metadata.get('chart_type', 'graphique')
    
    # Titre et source
    title = f"📊 **{chart_type.title()}**"
    source_info = f"*Source: {pdf_name}"
    if page_num != 'N/A':
        source_info += f", page {page_num}"
    source_info += "*"
    
    print(f"🎨 Tentative affichage: {title}")
    print(f"🔍 Métadonnées complètes: {metadata}")
    
    # RECHERCHE INTELLIGENTE DU CHEMIN DE L'IMAGE
    image_path = None
    
    # Méthode 1: Chercher dans les métadonnées
    possible_keys = ['image_path', 'source', 'file_path', 'path', 'image_file', 'filepath']
    for key in possible_keys:
        if key in metadata and metadata[key]:
            potential_path = str(metadata[key])
            print(f"🔍 Test métadonnée {key}: {potential_path}")
            
            # Tester le chemin direct
            if Path(potential_path).exists():
                image_path = potential_path
                print(f"✅ Image trouvée (chemin direct): {image_path}")
                break
            
            # Tester avec le dossier images/
            filename = Path(potential_path).name
            test_path = Path('images') / filename
            if test_path.exists():
                image_path = str(test_path)
                print(f"✅ Image trouvée (dossier images): {image_path}")
                break
    
    # Méthode 2: Construire le chemin à partir du PDF et de la page
    if not image_path and pdf_name and page_num != 'N/A':
        import glob
        
        print(f"🔍 Recherche par pattern PDF + page...")
        
        # Nettoyer le nom du PDF pour la recherche
        pdf_clean = pdf_name.replace('.pdf', '').replace(' ', '*')
        
        # Essayer différents patterns
        patterns = [
            f"images/{pdf_clean}*{page_num}*.png",
            f"images/*{pdf_clean}*{page_num}*.png",
            f"images/{pdf_clean}*p{page_num}*.png",
            f"images/*page*{page_num}*.png",
            f"images/*{page_num}*.png"
        ]
        
        for pattern in patterns:
            print(f"   Pattern: {pattern}")
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
                print(f"✅ Image trouvée par pattern: {image_path}")
                break
    
    # Méthode 3: Fallback - prendre n'importe quelle image de la page
    if not image_path and page_num != 'N/A':
        import glob
        fallback_pattern = f"images/*{page_num}*.png"
        matches = glob.glob(fallback_pattern)
        if matches:
            image_path = matches[0]
            print(f"✅ Image trouvée (fallback): {image_path}")
    
    # AFFICHAGE DE L'IMAGE
    displayed = False
    
    if image_path and Path(image_path).exists():
        try:
            if CHAINLIT_AVAILABLE:
                # Message avec titre et source
                await cl.Message(content=f"{title}\n\n{source_info}").send()
                
                # Afficher l'image
                elements = [cl.Image(name="chart", path=str(image_path), display="inline")]
                await cl.Message(content="", elements=elements).send()
                
                # Confirmation
                await cl.Message(content=f"✅ **Graphique affiché**\n📁 Fichier: `{Path(image_path).name}`").send()
                
                print(f"✅ IMAGE CHAINLIT AFFICHÉE: {image_path}")
                displayed = True
                
            elif STREAMLIT_AVAILABLE:
                st.markdown(f"### {title}")
                st.markdown(source_info)
                st.image(str(image_path), caption=f"Fichier: {Path(image_path).name}")
                print(f"✅ IMAGE STREAMLIT AFFICHÉE: {image_path}")
                displayed = True
                
            else:
                # Console
                print(f"\n{title}")
                print(source_info)
                print(f"🖼️ IMAGE DISPONIBLE: {image_path}")
                displayed = True
                
        except Exception as e:
            print(f"❌ Erreur affichage image: {e}")
    
    # Si pas d'image trouvée, afficher la description avec diagnostic
    if not displayed:
        print(f"⚠️ AUCUNE IMAGE TROUVÉE")
        
        diagnostic_msg = f"""
{title}

{source_info}

⚠️ **Graphique non affiché - Image manquante**

🔍 **Diagnostic:**
📁 PDF: `{pdf_name}`
📄 Page: `{page_num}`
🔍 Métadonnées recherchées: `{list(metadata.keys())}`

📝 **Description du graphique:**
{content}

💡 **Pour corriger:** Vérifiez que l'image existe dans le dossier `images/`
"""
        
        if CHAINLIT_AVAILABLE:
            await cl.Message(content=diagnostic_msg).send()
        elif STREAMLIT_AVAILABLE:
            st.markdown(diagnostic_msg)
        else:
            print(diagnostic_msg)
    
    return True


def format_table_for_display(content: str, max_width: int = 80) -> str:
    """Formate un tableau pour l'affichage."""
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return content
    
    # Limiter la largeur et le nombre de lignes
    formatted_lines = []
    for i, line in enumerate(lines[:10]):  # Max 10 lignes
        if len(line) > max_width:
            line = line[:max_width-3] + "..."
        formatted_lines.append(line)
    
    if len(lines) > 10:
        formatted_lines.append("... (tableau tronqué)")
    
    return '\n'.join(formatted_lines)

async def process_and_display_visual_elements(visual_elements: List[Dict[str, Any]], user_question: str) -> bool:
    """Traite et affiche tous les éléments visuels pertinents."""
    if not visual_elements:
        return False
    
    # AJOUTEZ CETTE LIGNE POUR LE DEBUG
    debug_visual_elements(visual_elements)
    
    print(f"🎨 Traitement de {len(visual_elements)} éléments visuels...")
    

    # Message d'introduction
    intro_msg = f"📊 **Contenu visuel ANSD pertinent**\n*En rapport avec: {user_question}*\n"
    
    if CHAINLIT_AVAILABLE:
        await cl.Message(content=intro_msg).send()
    elif STREAMLIT_AVAILABLE:
        st.markdown(intro_msg)
    else:
        print(intro_msg)
    
    displayed_count = 0
    
    for i, element in enumerate(visual_elements, 1):
        try:
            success = await display_visual_element(element, user_question)
            if success:
                displayed_count += 1
        except Exception as e:
            print(f"❌ Erreur affichage élément {i}: {e}")
            continue
    
    # Message de résumé
    if displayed_count > 0:
        summary = f"✅ **Affichage terminé**: {displayed_count} élément(s) visuel(s) affiché(s)"
        
        if CHAINLIT_AVAILABLE:
            await cl.Message(content=summary).send()
        elif STREAMLIT_AVAILABLE:
            st.success(summary)
        else:
            print(summary)
        
        return True
    
    return False

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def format_docs_with_metadata(docs) -> str:
    """Formatage avancé des documents avec métadonnées enrichies pour l'ANSD."""
    
    if not docs:
        return "❌ Aucun document pertinent trouvé dans la base de données ANSD."
    
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Extraction des métadonnées
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Informations sur la source
        source_info = []
        if 'source' in metadata:
            source_info.append(f"📄 Source: {metadata['source']}")
        if 'pdf_name' in metadata:
            source_info.append(f"📋 Document: {metadata['pdf_name']}")
        if 'page_num' in metadata:
            source_info.append(f"📖 Page: {metadata['page_num']}")
        if 'indexed_at' in metadata:
            source_info.append(f"🕐 Indexé: {metadata['indexed_at'][:10]}")
        
        # En-tête du document
        header = f"\n{'='*50}\n📊 DOCUMENT ANSD #{i}\n"
        if source_info:
            header += "\n".join(source_info) + "\n"
        header += f"{'='*50}\n"
        
        # Contenu avec nettoyage
        content = doc.page_content.strip()
        
        # Détecter le type de contenu
        content_indicators = []
        if any(keyword in content.lower() for keyword in ['rgph', 'recensement']):
            content_indicators.append("🏘️ RECENSEMENT")
        if any(keyword in content.lower() for keyword in ['eds', 'démographique']):
            content_indicators.append("👥 DÉMOGRAPHIE")
        if any(keyword in content.lower() for keyword in ['esps', 'pauvreté']):
            content_indicators.append("💰 PAUVRETÉ")
        if any(keyword in content.lower() for keyword in ['économie', 'pib']):
            content_indicators.append("📈 ÉCONOMIE")
        
        if content_indicators:
            header += f"🏷️ Catégories: {' | '.join(content_indicators)}\n{'-'*50}\n"
        
        formatted_parts.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted_parts)

# =============================================================================
# FONCTIONS PRINCIPALES AVEC SUPPORT VISUEL ET SUGGESTIONS
# =============================================================================

async def retrieve(state, *, config):
    """Fonction de récupération avec extraction des éléments visuels."""
    print("🔍 ---RETRIEVE AVEC SUPPORT VISUEL COMPLET---")
    
    # Gestion hybride dict/dataclass
    if isinstance(state, dict):
        messages = state.get("messages", [])
        print("📝 State reçu comme dictionnaire")
    else:
        messages = getattr(state, "messages", [])
        print("📝 State reçu comme dataclass")
    
    # Extraction de la question
    question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            question = msg.content
            break
    
    if not question:
        print("❌ Aucune question trouvée")
        return {"documents": [], "visual_elements": [], "has_visual_content": False}
    
    print(f"📝 Question: {question}")
    
    try:
        print("📄 Récupération documents textuels et visuels...")
        
        # Configuration Pinecone
        safe_config = dict(config) if config else {}
        if 'configurable' not in safe_config:
            safe_config['configurable'] = {}
        
        # Augmenter le nombre de documents récupérés pour capturer plus d'éléments visuels
        safe_search_kwargs = {
            "k": 20,  # Augmenté pour les éléments visuels
        }
        safe_config['configurable']['search_kwargs'] = safe_search_kwargs
        
        # Utilisation du retriever
        async with retrieval.make_retriever(safe_config) as retriever:
            documents = await retriever.ainvoke(question, safe_config)
            
            print(f"✅ Documents récupérés: {len(documents)}")
            
            # Séparer les documents textuels et visuels
            text_docs, visual_elements = extract_visual_elements(documents)
            
            print(f"📄 Documents textuels: {len(text_docs)}")
            print(f"🎨 Éléments visuels bruts: {len(visual_elements)}")
            
            # Analyser la pertinence des éléments visuels
            relevant_visual_elements = analyze_visual_relevance_enhanced(visual_elements, question)
            
            print(f"🎯 Éléments visuels pertinents: {len(relevant_visual_elements)}")
            
            return {
                "documents": text_docs,
                "visual_elements": relevant_visual_elements,
                "has_visual_content": len(relevant_visual_elements) > 0
            }
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        import traceback
        traceback.print_exc()
        return {"documents": [], "visual_elements": [], "has_visual_content": False}


async def generate(state, *, config):
    """Génération avec affichage automatique des éléments visuels ET suggestions de questions."""
    
    print("🤖 ---GENERATE AVEC VISUELS ET SUGGESTIONS COMPLET---")
    
    # Gestion hybride dict/dataclass
    if isinstance(state, dict):
        messages = state.get("messages", [])
        documents = state.get("documents", [])
        visual_elements = state.get("visual_elements", [])
        has_visual = state.get("has_visual_content", False)
        print("📝 State reçu comme dictionnaire")
    else:
        messages = getattr(state, "messages", [])
        documents = getattr(state, "documents", [])
        visual_elements = getattr(state, "visual_elements", [])
        has_visual = getattr(state, "has_visual_content", False)
        print("📝 State reçu comme dataclass")
    
    # Import des modules nécessaires
    try:
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return {"messages": [AIMessage(content="❌ Erreur de configuration ANSD.")], "documents": documents}
    
    # Extraire la question
    user_question = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            user_question = msg.content
            break
    
    print(f"❓ Question: {user_question}")
    print(f"📄 Documents disponibles: {len(documents)}")
    print(f"🎨 Éléments visuels disponibles: {len(visual_elements)}")
    
    try:
        # =============================================================================
        # ÉTAPE 1 : AFFICHER LES ÉLÉMENTS VISUELS PERTINENTS
        # =============================================================================
        
        if has_visual and visual_elements:
            print("\n🎨 ÉTAPE 1 : Affichage des éléments visuels...")
            await process_and_display_visual_elements(visual_elements, user_question)
        
        # =============================================================================
        # ÉTAPE 2 : ESSAYER AVEC LES DOCUMENTS INDEXÉS
        # =============================================================================
        
        print("\n🔍 ÉTAPE 2 : Recherche dans les documents indexés...")
        
        if documents:
            # Prompt adapté pour mentionner les éléments visuels
            visual_context = ""
            if has_visual:
                visual_context = f"\n\nNote: {len(visual_elements)} élément(s) visuel(s) (graphiques/tableaux) ont été affichés ci-dessus en rapport avec cette question. Référencez-les dans votre réponse si pertinents."
            
            prompt_documents_only = ChatPromptTemplate.from_messages([
                ("system", f"""Vous êtes un expert statisticien de l'ANSD. 

Analysez les documents fournis et répondez à la question en utilisant EXCLUSIVEMENT ces documents.

RÈGLES :
✅ Utilisez TOUTES les informations pertinentes trouvées dans les documents
✅ Donnez les chiffres EXACTS trouvés
✅ Citez les sources précises (nom du document, page)
✅ Développez votre réponse avec les détails disponibles
✅ Si vous trouvez des informations partielles, présentez-les clairement

❌ Seulement si VRAIMENT AUCUNE information pertinente n'est trouvée, répondez : "INFORMATION NON DISPONIBLE"

FORMAT DE RÉPONSE :

**RÉPONSE DIRECTE :**
[Réponse détaillée basée sur les documents trouvés]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeurs des documents]
- Année de référence : [année des documents]
- Source : [nom exact du document, page X]
- Méthodologie : [enquête/recensement des documents]

**CONTEXTE ADDITIONNEL :**
[Toutes les informations complémentaires trouvées dans les documents]

{visual_context}

DOCUMENTS DISPONIBLES :
{{context}}"""),
                ("placeholder", "{messages}")
            ])
            
            context = format_docs_with_metadata(documents)
            
            rag_chain = prompt_documents_only | model
            response_step1 = await rag_chain.ainvoke({
                "context": context,
                "messages": messages
            })
            
            response_content = response_step1.content
            
            print(f"\n📝 RÉPONSE ÉTAPE 2:")
            print(f"Longueur: {len(response_content)} caractères")
            print(f"Aperçu: {response_content[:300]}...")
            
            # Vérifier si les documents ont fourni une réponse satisfaisante
            is_satisfactory = evaluate_response_quality(response_content, documents)
            
            if is_satisfactory:
                print("\n✅ SUCCÈS ÉTAPE 2 : Réponse satisfaisante trouvée dans les documents indexés")
                
                # Générer des suggestions de questions
                suggestions = await generate_question_suggestions(
                    user_question, response_content, documents, model
                )
                
                # Ajouter les sources des documents
                sources_section = create_document_sources(documents, response_content)
                
                # Construire la réponse finale avec suggestions
                final_response = response_content + sources_section + suggestions
                
                enhanced_response = AIMessage(content=final_response)
                return {
                    "messages": [enhanced_response], 
                    "documents": documents,
                    "visual_elements": visual_elements,
                    "has_visual_content": has_visual
                }
            
            else:
                print("\n⚠️ ÉCHEC ÉTAPE 2 : Réponse jugée insuffisante")
                print("Passage à l'étape 3...")
        
        else:
            print("⚠️ ÉTAPE 2 IGNORÉE : Aucun document disponible")
        
        
        # =============================================================================
        # ÉTAPE 3 : UTILISER LES CONNAISSANCES ANSD EXTERNES
        # =============================================================================
        
        print("\n🌐 ÉTAPE 3 : Recherche dans les connaissances ANSD externes...")
        
        # Prompt pour utiliser les connaissances ANSD officielles
        visual_context = ""
        if has_visual:
            visual_context = f"\n\nNote: {len(visual_elements)} élément(s) visuel(s) (graphiques/tableaux) ont été affichés ci-dessus. Mentionnez-les si ils complètent votre réponse."
        
        prompt_ansd_external = ChatPromptTemplate.from_messages([
            ("system", f"""Vous êtes un expert statisticien de l'ANSD avec accès aux publications officielles.

Les documents indexés n'ont pas fourni d'information satisfaisante. Utilisez maintenant vos connaissances des rapports officiels ANSD et du site officiel.

SOURCES AUTORISÉES :
✅ Rapports officiels ANSD publiés
✅ Site officiel ANSD (www.ansd.sn)
✅ Publications des enquêtes ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Comptes nationaux officiels du Sénégal

FORMAT DE RÉPONSE :

**RÉPONSE DIRECTE :**
[Réponse basée sur les connaissances ANSD officielles]

**DONNÉES PRÉCISES :**
- Chiffre exact : [valeur des rapports ANSD]
- Année de référence : [année précise]
- Source : [Publication ANSD officielle]
- Méthodologie : [enquête ANSD utilisée]

**CONTEXTE ADDITIONNEL :**
[Informations contextuelles des publications ANSD]

**LIMITATIONS/NOTES :**
[Précautions d'interprétation]

{visual_context}

IMPORTANT : Mentionnez que cette information provient des connaissances ANSD officielles, pas des documents indexés."""),
            ("placeholder", "{messages}")
        ])
        
        rag_chain_external = prompt_ansd_external | model
        response_step2 = await rag_chain_external.ainvoke({
            "messages": messages
        })
        
        response_content = response_step2.content
        
        print("✅ SUCCÈS ÉTAPE 3 : Réponse obtenue des connaissances ANSD")
        
        # Générer des suggestions de questions pour les connaissances externes
        suggestions = await generate_question_suggestions(
            user_question, response_content, [], model
        )
        
        # Ajouter les sources externes
        sources_section = create_external_ansd_sources(response_content)
        
        # Construire la réponse finale avec suggestions
        final_response = response_content + sources_section + suggestions
        
        enhanced_response = AIMessage(content=final_response)
        return {
            "messages": [enhanced_response], 
            "documents": documents,
            "visual_elements": visual_elements,
            "has_visual_content": has_visual
        }
    
    except Exception as e:
        print(f"❌ ERREUR GÉNÉRATION: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback final
        fallback_response = AIMessage(content=
            "❌ Informations non disponibles dans les documents indexés et les sources ANSD consultées. "
            "Veuillez consulter directement l'ANSD (www.ansd.sn) pour cette information spécifique."
        )
        return {
            "messages": [fallback_response], 
            "documents": documents,
            "visual_elements": visual_elements,
            "has_visual_content": has_visual
        }

# =============================================================================
# FONCTIONS DE SUGGESTIONS DE QUESTIONS
# =============================================================================

async def generate_question_suggestions(user_question, response_content, documents, model):
    """Génère des suggestions de questions suivantes contextuelles."""
    
    print("\n🔮 GÉNÉRATION DES SUGGESTIONS DE QUESTIONS...")
    
    try:
        # Analyser le contexte pour les suggestions
        document_topics = extract_topics_from_documents(documents) if documents else []
        response_topics = extract_topics_from_response(response_content)
        
        # Prompt pour générer des suggestions contextuelles
        suggestions_prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert ANSD qui aide les utilisateurs à explorer les statistiques du Sénégal.

Basé sur la question posée et la réponse fournie, générez 4 questions de suivi pertinentes et spécifiques au contexte ANSD.

RÈGLES POUR LES SUGGESTIONS :
✅ Questions COMPLÉMENTAIRES à la question originale
✅ Utilisez la terminologie ANSD (RGPH, EDS, ESPS, EHCVM, ENES)
✅ Questions spécifiques au Sénégal et aux données disponibles
✅ Mélangez différents angles : temporel, géographique, thématique, méthodologique
✅ Questions qui approfondissent ou élargissent le sujet
✅ Évitez de répéter la question originale

TYPES DE QUESTIONS À PRIVILÉGIER :
🔍 Comparaisons temporelles (évolution, tendances)
🗺️ Analyses géographiques (régions, départements)
👥 Segmentations démographiques (âge, sexe, milieu)
📊 Indicateurs connexes ou complémentaires
🔬 Aspects méthodologiques des enquêtes
💡 Implications politiques ou sociales

FORMAT EXACT :
**❓ QUESTIONS SUGGÉRÉES :**

1. [Question sur l'évolution temporelle ou comparaison entre périodes]

2. [Question sur la répartition géographique ou variations régionales]

3. [Question sur un indicateur connexe ou complémentaire]

4. [Question méthodologique ou d'approfondissement thématique]

CONTEXTE QUESTION ORIGINALE :
{original_question}

THÈMES IDENTIFIÉS DANS LA RÉPONSE :
{response_topics}

THÈMES DISPONIBLES DANS LES DOCUMENTS :
{document_topics}"""),
            ("user", "Générez maintenant 4 suggestions de questions de suivi pertinentes.")
        ])
        
        # Préparer le contexte pour les suggestions
        context_data = {
            "original_question": user_question,
            "response_topics": ", ".join(response_topics) if response_topics else "Analyse générale",
            "document_topics": ", ".join(document_topics) if document_topics else "Documents généraux ANSD"
        }
        
        # Générer les suggestions
        suggestions_chain = suggestions_prompt | model
        suggestions_response = await suggestions_chain.ainvoke(context_data)
        
        suggestions_content = suggestions_response.content
        
        print(f"✅ Suggestions générées: {len(suggestions_content)} caractères")
        
        return f"\n\n{suggestions_content}"
        
    except Exception as e:
        print(f"❌ Erreur génération suggestions: {e}")
        
        # Suggestions de fallback basiques
        fallback_suggestions = generate_fallback_suggestions(user_question)
        return f"\n\n{fallback_suggestions}"

def extract_topics_from_documents(documents):
    """Extrait les thèmes principaux des documents."""
    
    if not documents:
        return []
    
    topics = set()
    
    # Mots-clés thématiques ANSD
    ansd_keywords = {
        'démographie': ['population', 'habitants', 'démographique', 'natalité', 'mortalité'],
        'économie': ['économie', 'pib', 'revenus', 'emploi', 'secteur'],
        'éducation': ['éducation', 'scolarisation', 'alphabétisation', 'école'],
        'santé': ['santé', 'maternelle', 'vaccination', 'morbidité'],
        'pauvreté': ['pauvreté', 'pauvre', 'indigence', 'vulnérabilité'],
        'géographie': ['région', 'département', 'urbain', 'rural', 'dakar'],
        'enquêtes': ['rgph', 'eds', 'esps', 'ehcvm', 'enes', 'recensement']
    }
    
    # Analyser le contenu des documents
    combined_content = " ".join([doc.page_content.lower() for doc in documents if hasattr(doc, 'page_content')])
    
    for theme, keywords in ansd_keywords.items():
        if any(keyword in combined_content for keyword in keywords):
            topics.add(theme)
    
    return list(topics)

def extract_topics_from_response(response_content):
    """Extrait les thèmes principaux de la réponse."""
    
    topics = []
    response_lower = response_content.lower()
    
    # Détection de thèmes spécifiques
    if any(term in response_lower for term in ['population', 'habitants', 'démographique']):
        topics.append('démographie')
    
    if any(term in response_lower for term in ['économie', 'pib', 'croissance', 'secteur']):
        topics.append('économie')
    
    if any(term in response_lower for term in ['pauvreté', 'pauvre', 'indigence']):
        topics.append('pauvreté')
    
    if any(term in response_lower for term in ['emploi', 'travail', 'chômage']):
        topics.append('emploi')
    
    if any(term in response_lower for term in ['éducation', 'école', 'scolarisation']):
        topics.append('éducation')
    
    if any(term in response_lower for term in ['santé', 'mortalité', 'morbidité']):
        topics.append('santé')
    
    if any(term in response_lower for term in ['région', 'département', 'géographique']):
        topics.append('géographie')
    
    return topics

def generate_fallback_suggestions(user_question):
    """Génère des suggestions de base si l'IA échoue."""
    
    question_lower = user_question.lower()
    
    # Suggestions basées sur le contenu de la question
    if any(term in question_lower for term in ['population', 'habitants']):
        return """**❓ QUESTIONS SUGGÉRÉES :**

1. Quelle est l'évolution de la population sénégalaise entre les différents recensements ?

2. Comment la population se répartit-elle entre les régions du Sénégal ?

3. Quels sont les indicateurs démographiques clés (taux de natalité, mortalité) ?

4. Quelle est la répartition de la population par groupes d'âge et par sexe ?"""
    
    elif any(term in question_lower for term in ['pauvreté', 'pauvre']):
        return """**❓ QUESTIONS SUGGÉRÉES :**

1. Comment le taux de pauvreté a-t-il évolué au Sénégal ces dernières années ?

2. Quelles sont les régions les plus touchées par la pauvreté ?

3. Quels sont les profils des ménages pauvres selon l'ESPS ?

4. Quelles sont les stratégies gouvernementales de lutte contre la pauvreté ?"""
    
    elif any(term in question_lower for term in ['emploi', 'travail']):
        return """**❓ QUESTIONS SUGGÉRÉES :**

1. Quelle est l'évolution du taux de chômage au Sénégal ?

2. Comment l'emploi se répartit-il par secteur d'activité ?

3. Quels sont les défis de l'emploi des jeunes selon l'ENES ?

4. Quelle est la part de l'emploi informel dans l'économie sénégalaise ?"""
    
    else:
        # Suggestions génériques
        return """**❓ QUESTIONS SUGGÉRÉES :**

1. Quels sont les derniers résultats du RGPH-5 sur la population sénégalaise ?

2. Comment les indicateurs sociaux ont-ils évolué selon les enquêtes ANSD ?

3. Quelles sont les principales disparités régionales observées ?

4. Quels défis méthodologiques pose la collecte de données au Sénégal ?"""

# =============================================================================
# FONCTIONS D'ÉVALUATION ET DE SOURCES
# =============================================================================

def evaluate_response_quality(response_content, documents):
    """Évaluation CORRIGÉE - Moins stricte pour accepter les réponses des documents."""
    
    response_lower = response_content.lower()
    
    print(f"🔍 ÉVALUATION DE LA RÉPONSE:")
    print(f"   Longueur: {len(response_content)} caractères")
    print(f"   Aperçu: {response_content[:150]}...")
    
    # Critères d'échec STRICTS (réponse clairement insuffisante)
    failure_indicators = [
        "information non disponible",
        "cette information n'est pas disponible",
        "aucune information",
        "pas disponible dans les documents",
        "impossible de répondre",
        "données non présentes",
        "ne peut pas répondre",
        "informations insuffisantes"
    ]
    
    # Si contient un indicateur d'échec EXPLICITE
    for indicator in failure_indicators:
        if indicator in response_lower:
            print(f"   ❌ ÉCHEC: Contient '{indicator}'")
            return False
    
    # Critères de succès ASSOUPLIS
    success_indicators = {
        'has_numbers': bool(re.search(r'\d+', response_content)),
        'has_content': len(response_content) > 100,
        'has_words': len(response_content.split()) > 20,
        'has_structure': '**' in response_content or '-' in response_content,
        'mentions_documents': any(term in response_lower for term in ['chapitre', 'page', 'document', 'source']),
        'has_specific_content': any(term in response_lower for term in [
            'répartition', 'secteur', 'occupés', 'institutionnel', 'emploi', 'travail',
            'population', 'habitants', 'taux', 'pourcentage', 'statistique'
        ])
    }
    
    print(f"   📊 Critères détaillés:")
    for criterion, passed in success_indicators.items():
        status = "✅" if passed else "❌"
        print(f"      {status} {criterion}")
    
    success_count = sum(success_indicators.values())
    
    # SEUIL RÉDUIT : 2 critères au lieu de 3
    is_satisfactory = success_count >= 2
    
    print(f"   📈 Score: {success_count}/6 critères")
    print(f"   🎯 Résultat: {'✅ SATISFAISANT' if is_satisfactory else '❌ INSUFFISANT'}")
    
    return is_satisfactory

def create_document_sources(documents, response_content):
    """Crée la section sources pour les documents indexés."""
    
    sources_section = "\n\n📚 **Sources utilisées (Documents indexés):**\n"
    
    # Extraire les documents réellement pertinents
    relevant_docs = []
    response_lower = response_content.lower()
    
    for doc in documents:
        if hasattr(doc, 'page_content'):
            doc_content = doc.page_content.lower()
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Vérifier si le document a été utilisé (overlap de contenu)
            doc_words = set(doc_content.split())
            response_words = set(response_lower.split())
            
            # Mots significatifs communs
            significant_words = {word for word in doc_words.intersection(response_words) 
                               if len(word) > 4}
            
            # Si overlap significatif OU données numériques communes
            doc_numbers = set(re.findall(r'\d+[.,]?\d*', doc_content))
            response_numbers = set(re.findall(r'\d+[.,]?\d*', response_lower))
            
            if len(significant_words) > 2 or doc_numbers.intersection(response_numbers):
                doc_name = metadata.get('pdf_name', 'Document ANSD')
                page_num = metadata.get('page_num', 'Non spécifiée')
                
                if '/' in doc_name:
                    doc_name = doc_name.split('/')[-1]
                
                formatted = f"{doc_name}, page {page_num}" if page_num != 'Non spécifiée' else doc_name
                relevant_docs.append(formatted)
    
    # Ajouter les sources ou fallback
    if relevant_docs:
        for doc in relevant_docs[:5]:  # Max 5 sources
            sources_section += f"• {doc}\n"
    else:
        # Si aucun document spécifique identifié, utiliser tous
        for doc in documents[:3]:  # Max 3 sources
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_name = metadata.get('pdf_name', 'Document ANSD')
            page_num = metadata.get('page_num', 'Non spécifiée')
            
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            
            formatted = f"{doc_name}, page {page_num}" if page_num != 'Non spécifiée' else doc_name
            sources_section += f"• {formatted}\n"
    
    return sources_section

def create_external_ansd_sources(response_content):
    """Crée la section sources pour les connaissances ANSD externes."""
    
    sources_section = "\n\n📚 **Sources officielles :**\n"
    
    response_lower = response_content.lower()
    
    # Détecter les sources spécifiques mentionnées dans la réponse
    detected_sources = []
    
    # Enquêtes spécifiques
    if 'ehcvm' in response_lower or 'conditions de vie' in response_lower:
        detected_sources.append("• ANSD - Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM), 2018-2019")
    
    if 'esps' in response_lower or 'pauvreté' in response_lower:
        detected_sources.append("• ANSD - Enquête de Suivi de la Pauvreté au Sénégal (ESPS), 2018-2019")
    
    if 'eds' in response_lower or 'démographique et santé' in response_lower:
        detected_sources.append("• ANSD - Enquête Démographique et de Santé (EDS), 2023")
    
    if 'rgph' in response_lower or 'recensement' in response_lower:
        detected_sources.append("• ANSD - Recensement Général de la Population et de l'Habitat (RGPH), 2023")
    
    if 'enes' in response_lower or 'emploi' in response_lower:
        detected_sources.append("• ANSD - Enquête Nationale sur l'Emploi au Sénégal (ENES), 2021")
    
    # Publications économiques
    if 'pib' in response_lower or 'comptes nationaux' in response_lower:
        detected_sources.append("• ANSD - Comptes Nationaux du Sénégal, 2023")
    
    if 'prix' in response_lower or 'inflation' in response_lower:
        detected_sources.append("• ANSD - Indices des Prix à la Consommation, 2024")
    
    # Toujours ajouter le site officiel
    detected_sources.append("• Site officiel ANSD (www.ansd.sn)")
    
    # Ajouter note explicative
    sources_section += "• **Note :** Informations issues des connaissances des publications ANSD officielles\n"
    
    # Ajouter les sources détectées (max 4 pour éviter la surcharge)
    for source in detected_sources[:4]:
        sources_section += f"{source}\n"
    
    return sources_section

# =============================================================================
# CONFIGURATION DU WORKFLOW AVEC SUPPORT VISUEL COMPLET
# =============================================================================

workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "ImprovedSimpleRagWithVisualsAndSuggestions"