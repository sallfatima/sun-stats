"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""
import os
from typing import Optional, List, Dict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pinecone import Index, Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from typing import List, Dict, Any
from datetime import datetime


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

def load_pinecone_index(index_name: str):
    """
    Charge un index Pinecone existant, ou le crée automatiquement s’il n’existe pas.
    """
    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    indexes = pinecone_client.list_indexes().names()
    print("🔎 Index disponibles :", indexes)

    if index_name not in indexes:
        print(f"⚠️ L'index '{index_name}' n'existe pas. Création...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec= ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"
            )
        )
        print(f"✅ Index '{index_name}' créé.")

    return pinecone_client.Index(index_name)

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)
# =============================================================================
# FICHIER 4: src/shared/utils.py
# =============================================================================
# AJOUTEZ CES FONCTIONS À LA FIN DE VOTRE FICHIER EXISTANT (ne pas remplacer le contenu existant)

# =============================================================================
# FONCTIONS UTILITAIRES AMÉLIORÉES POUR L'ANSD
# =============================================================================

import re
from typing import List, Dict, Any
from datetime import datetime

def detect_ansd_survey_type(text: str) -> List[str]:
    """Détecte le type d'enquête ANSD dans un texte."""
    
    text_lower = text.lower()
    surveys = []
    
    # Dictionnaire des enquêtes avec leurs variantes
    survey_patterns = {
        "rgph": ["rgph", "recensement", "population", "habitat", "recensement général"],
        "eds": ["eds", "enquête démographique", "santé", "demographic health survey"],
        "esps": ["esps", "pauvreté", "enquête de suivi", "poverty"],
        "ehcvm": ["ehcvm", "conditions de vie", "ménages", "budget", "consommation"],
        "enes": ["enes", "emploi", "chômage", "activité économique"],
        "comptes_nationaux": ["pib", "comptes nationaux", "économie", "croissance"],
    }
    
    for survey, patterns in survey_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            surveys.append(survey)
    
    return surveys

def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
    """Extrait les données numériques d'un texte avec leur contexte."""
    
    numerical_data = []
    
    # Patterns pour différents types de données
    patterns = [
        # Pourcentages
        (r'(\d+(?:[.,]\d+)?)\s*%', 'percentage'),
        # Nombres avec unités monétaires
        (r'(\d+(?:[.,]\d+)?)\s*(fcfa|francs?|f\s*cfa)', 'currency'),
        # Populations en millions/milliards
        (r'(\d+(?:[.,]\d+)?)\s*(millions?|milliards?)', 'population'),
        # Années
        (r'\b(20\d{2})\b', 'year'),
        # Nombres simples avec contexte
        (r'(\d+(?:[.,]\d+)?)', 'number'),
    ]
    
    for pattern, data_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extraire le contexte autour du nombre (50 caractères avant et après)
            start_context = max(0, match.start() - 50)
            end_context = min(len(text), match.end() + 50)
            context = text[start_context:end_context].strip()
            
            numerical_data.append({
                'value': match.group(1),
                'type': data_type,
                'context': context,
                'position': match.span()
            })
    
    return numerical_data

def format_ansd_metadata(metadata: Dict[str, Any]) -> str:
    """Formate les métadonnées spécifiques aux documents ANSD."""
    
    formatted_parts = []
    
    # Source principale
    if 'source' in metadata:
        formatted_parts.append(f"📄 Source: {metadata['source']}")
    
    # Nom du document
    if 'pdf_name' in metadata:
        formatted_parts.append(f"📋 Document: {metadata['pdf_name']}")
    
    # Numéro de page
    if 'page_num' in metadata:
        formatted_parts.append(f"📖 Page: {metadata['page_num']}")
    
    # Date d'indexation
    if 'indexed_at' in metadata:
        date_str = metadata['indexed_at'][:10] if len(metadata['indexed_at']) >= 10 else metadata['indexed_at']
        formatted_parts.append(f"🕐 Indexé: {date_str}")
    
    # Type de document
    if 'type' in metadata:
        formatted_parts.append(f"📊 Type: {metadata['type']}")
    
    # Taille du fichier (si disponible)
    if 'file_size' in metadata:
        size_mb = metadata['file_size'] / (1024 * 1024)
        formatted_parts.append(f"💾 Taille: {size_mb:.1f} MB")
    
    return " | ".join(formatted_parts) if formatted_parts else "📄 Métadonnées non disponibles"

def calculate_document_relevance_score(doc_content: str, query: str, survey_weights: Dict[str, float] = None) -> float:
    """Calcule un score de pertinence pour un document par rapport à une requête."""
    
    if not doc_content or not query:
        return 0.0
    
    doc_lower = doc_content.lower()
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    score = 0.0
    
    # Score basé sur les mots-clés de la requête
    for word in query_words:
        if len(word) > 3:  # Ignorer les mots très courts
            word_count = doc_lower.count(word)
            score += word_count * 2
    
    # Bonus pour les termes ANSD spécifiques
    ansd_terms = {
        'rgph': 5,
        'eds': 5,
        'esps': 5,
        'ehcvm': 5,
        'enes': 5,
        'ansd': 3,
        'sénégal': 3,
        'recensement': 4,
        'enquête': 3,
        'statistique': 3
    }
    
    for term, weight in ansd_terms.items():
        if term in doc_lower:
            score += weight
    
    # Bonus pour les données numériques
    numerical_patterns = [
        r'\d+[.,]\d+\s*%',  # Pourcentages
        r'\d+\s*(millions?|milliards?)',  # Grandes populations
        r'\d{4}',  # Années
    ]
    
    for pattern in numerical_patterns:
        matches = len(re.findall(pattern, doc_lower))
        score += matches * 2
    
    # Appliquer les poids des enquêtes si fournis
    if survey_weights:
        surveys = detect_ansd_survey_type(doc_content)
        for survey in surveys:
            if survey in survey_weights:
                score *= survey_weights[survey]
    
    return score

def clean_ansd_text(text: str) -> str:
    """Nettoie le texte des documents ANSD pour améliorer la lisibilité."""
    
    if not text:
        return ""
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Corriger les problèmes d'encodage courants
    replacements = {
        'Ã ': 'à ',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã§': 'ç',
        'Ã´': 'ô',
        'Ã¢': 'â',
        'Ã®': 'î',
        'Ã»': 'û',
        'Ã¹': 'ù',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text.strip()

def extract_ansd_indicators(text: str) -> List[Dict[str, str]]:
    """Extrait les indicateurs statistiques spécifiques à l'ANSD."""
    
    indicators = []
    
    # Patterns pour différents types d'indicateurs
    indicator_patterns = [
        # Taux démographiques
        (r'taux\s+(?:de\s+)?(?:natalité|mortalité|fécondité|croissance)\s*:\s*([^.\n]+)', 'démographique'),
        # Indicateurs économiques
        (r'(?:pib|produit\s+intérieur\s+brut)\s*:\s*([^.\n]+)', 'économique'),
        # Indicateurs de pauvreté
        (r'(?:taux\s+de\s+pauvreté|incidence\s+de\s+la\s+pauvreté)\s*:\s*([^.\n]+)', 'pauvreté'),
        # Indicateurs d'éducation
        (r'taux\s+(?:d\s*\'?\s*alphabétisation|de\s+scolarisation)\s*:\s*([^.\n]+)', 'éducation'),
        # Indicateurs de santé
        (r'(?:espérance\s+de\s+vie|mortalité\s+(?:infantile|maternelle))\s*:\s*([^.\n]+)', 'santé'),
    ]
    
    for pattern, category in indicator_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            indicators.append({
                'category': category,
                'indicator': match.group(0),
                'value': match.group(1).strip(),
                'position': match.span()
            })
    
    return indicators

def validate_ansd_response(response: str) -> Dict[str, Any]:
    """Valide qu'une réponse contient les éléments requis pour l'ANSD."""
    
    validation = {
        'has_numerical_data': False,
        'has_source_citation': False,
        'has_year_reference': False,
        'has_ansd_terminology': False,
        'quality_score': 0.0,
        'suggestions': []
    }
    
    response_lower = response.lower()
    
    # Vérifier la présence de données numériques
    if re.search(r'\d+(?:[.,]\d+)?(?:\s*%|\s*millions?|\s*milliards?)', response):
        validation['has_numerical_data'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Ajouter des données chiffrées précises")
    
    # Vérifier les citations de sources
    if any(term in response_lower for term in ['source:', 'rgph', 'eds', 'esps', 'ehcvm', 'enes', 'ansd']):
        validation['has_source_citation'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Citer les sources ANSD spécifiques")
    
    # Vérifier les références temporelles
    if re.search(r'20\d{2}|année\s+de\s+référence', response):
        validation['has_year_reference'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Préciser l'année de référence des données")
    
    # Vérifier la terminologie ANSD
    ansd_terms = ['enquête', 'recensement', 'indicateur', 'statistique', 'démographique', 'sénégal']
    if any(term in response_lower for term in ansd_terms):
        validation['has_ansd_terminology'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Utiliser la terminologie statistique appropriée")
    
    return validation

# =============================================================================
# FONCTIONS DE LOGGING SPÉCIALISÉES
# =============================================================================

def log_ansd_retrieval(query: str, documents: List, processing_time: float = None):
    """Log spécialisé pour le processus de récupération ANSD."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🕐 ANSD RETRIEVAL LOG - {timestamp}")
    print(f"{'='*60}")
    print(f"📝 Requête: {query}")
    print(f"📚 Documents trouvés: {len(documents)}")
    
    if processing_time:
        print(f"⏱️  Temps de traitement: {processing_time:.2f}s")
    
    # Analyser les types d'enquêtes trouvées
    survey_counts = {}
    for doc in documents:
        surveys = detect_ansd_survey_type(doc.page_content)
        for survey in surveys:
            survey_counts[survey] = survey_counts.get(survey, 0) + 1
    
    if survey_counts:
        print(f"📊 Types d'enquêtes identifiées:")
        for survey, count in survey_counts.items():
            print(f"   • {survey.upper()}: {count} document(s)")
    
    print(f"{'='*60}\n")

def log_ansd_generation(response: str, validation_result: Dict = None):
    """Log spécialisé pour le processus de génération ANSD."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🤖 ANSD GENERATION LOG - {timestamp}")
    print(f"{'='*60}")
    print(f"📏 Longueur de la réponse: {len(response)} caractères")
    
    if validation_result:
        print(f"⭐ Score de qualité: {validation_result['quality_score']}/100")
        if validation_result['suggestions']:
            print(f"💡 Suggestions d'amélioration:")
            for suggestion in validation_result['suggestions']:
                print(f"   • {suggestion}")
    
    # Extraire les données numériques mentionnées
    numerical_data = extract_numerical_data(response)
    if numerical_data:
        print(f"🔢 Données numériques détectées: {len(numerical_data)}")
        for data in numerical_data[:3]:  # Afficher les 3 premières
            print(f"   • {data['value']} ({data['type']})")
    
    print(f"{'='*60}\n")