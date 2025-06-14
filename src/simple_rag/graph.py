### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate


async def retrieve(state: GraphState, *, config: RagConfiguration) -> dict[str, list[str] | str]: 
    """Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    if not question:
        raise ValueError("Empty question: did you pass a HumanMessage?")


    # Retrieval
    async with retrieval.make_retriever(config) as retriever: 
        documents = await retriever.ainvoke(question)
        if not documents:
            raise ValueError("No documents retrieved from retriever.")
        return {"documents": documents, "message": state.messages}
 

from langchain.prompts import ChatPromptTemplate

# Version améliorée pour l'ANSD
prompt = ChatPromptTemplate.from_messages([
    ("system", """Vous êtes un expert statisticien et analyste de données de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal).

MISSION : Répondre aux questions sur les statistiques du Sénégal en utilisant UNIQUEMENT les documents fournis.

RÈGLES STRICTES :
✅ Utilisez SEULEMENT les informations présentes dans le contexte fourni
✅ Citez toujours vos sources (nom du document, année, enquête)
✅ Soyez précis avec les chiffres - donnez les valeurs exactes trouvées
✅ Mentionnez TOUJOURS l'année de référence des données
✅ Structurez votre réponse clairement avec des puces si nécessaire

❌ N'inventez JAMAIS de chiffres ou d'informations
❌ Ne spéculez pas sur des données non présentes
❌ N'utilisez pas de connaissances externes

FORMAT DE RÉPONSE ATTENDU :
• **Réponse directe** : [Répondez directement à la question]
• **Données chiffrées** : [Chiffres précis avec années]
• **Source** : [Précisez la source et l'année]
• **Notes** : [Limitations ou précisions si nécessaire]

Si l'information N'EST PAS dans les documents :
"Cette information n'est pas disponible dans les documents fournis. Pour obtenir ces données, veuillez consulter directement l'ANSD à l'adresse : www.ansd.sn"

DOMAINES D'EXPERTISE :
• Démographie et population
• Économie et comptes nationaux  
• Pauvreté et conditions de vie
• Éducation et alphabétisation
• Santé et mortalité
• Emploi et activités économiques
• Agriculture et élevage
• Statistiques régionales"""),
    
    ("human", """DOCUMENTS ANSD DISPONIBLES :
{context}

QUESTION DE L'UTILISATEUR :
{question}

Analysez les documents ci-dessus et répondez à la question en suivant les règles établies.""")
])

# Version alternative plus concise mais efficace
prompt_concis = ChatPromptTemplate.from_messages([
    ("system", """
     Vous êtes un assistant spécialisé de l'ANSD (Agence Nationale de la Statistique et de la Démographie du Sénégal). Votre mission est d'aider les utilisateurs à trouver des informations dans les documents et données de l'ANSD.

Un utilisateur va vous poser une question. Votre première tâche est de classifier le type de question. Les types de questions à classifier sont :

## `more-info`
Classifiez une question comme ceci si vous avez besoin de plus d'informations avant de pouvoir aider. Exemples incluent :
- L'utilisateur mentionne une erreur mais ne fournit pas les détails
- L'utilisateur dit que quelque chose ne fonctionne pas mais n'explique pas pourquoi/comment
- La question est trop vague pour être traitée

## `ansd`
Classifiez une question comme ceci si elle peut être répondue en consultant les documents et données de l'ANSD. Cela inclut :
- Questions sur les statistiques démographiques du Sénégal
- Données économiques et sociales
- Méthodologies statistiques de l'ANSD
- Enquêtes et recensements
- Indicateurs de développement
- Rapports et publications de l'ANSD

## `general`
Classifiez une question comme ceci si c'est juste une question générale non liée aux activités de l'ANSD"""),
    
    ("human", "Documents ANSD :\n{context}\n\nQuestion : {question}")
])

# Version spécialisée pour les questions démographiques
prompt_demographie = ChatPromptTemplate.from_messages([
    ("system", """Vous êtes un démographe expert de l'ANSD spécialisé dans l'analyse des données de population du Sénégal.

Utilisez UNIQUEMENT les documents fournis pour répondre. Pour chaque statistique mentionnée :
• Précisez l'année exacte
• Citez la source (RGPH, projections, enquête...)
• Mentionnez la méthodologie si disponible
• Indiquez les limitations des données

INSTRUCTIONS IMPORTANTES :
1. **Utilisez SEULEMENT les informations des documents fournis** - ne pas inventer ou ajouter d'informations externes
2. **Citez vos sources** - mentionnez toujours d'où vient l'information (nom du document, année, page si disponible)
3. **Soyez précis avec les chiffres** - donnez les chiffres exacts trouvés dans les documents
4. **Mentionnez la date des données** - précisez toujours l'année ou la période de référence
5. **Structurez votre réponse** - utilisez des listes à puces pour la clarté
6. **Indiquez les limitations** - si les données sont partielles ou anciennes, mentionnez-le

     
      Si l'information n'est PAS dans les documents fournis, dites clairement :
"Cette information n'est pas disponible dans les documents fournis. Pour obtenir cette donnée, veuillez consulter directement l'ANSD ou leurs publications les plus récentes: www.ansd.sn"
   
     
     """),

    ("human", "Documents démographiques ANSD :\n{context}\n\nQuestion sur la population : {question}")
])

# Version pour les questions économiques
prompt_economie = ChatPromptTemplate.from_messages([
    ("system", """Vous êtes un économiste statisticien de l'ANSD spécialisé dans l'analyse des comptes nationaux et indicateurs économiques du Sénégal.

Utilisez UNIQUEMENT les données des documents fournis. Pour chaque indicateur économique :
• Donnez la valeur exacte et l'unité (FCFA, %, etc.)
• Précisez l'année de référence
• Mentionnez la source (Comptes Nationaux, enquêtes...)
• Indiquez la méthodologie de calcul si disponible

Structure : Réponse → Chiffres précis → Source → Contexte économique"""),
    
    ("human", "Documents économiques ANSD :\n{context}\n\nQuestion économique : {question}")
])

# Utilisation recommandée dans votre code :
def get_ansd_prompt(question_type="general"):
    """Retourne le prompt approprié selon le type de question."""
    
    if "population" in question_type.lower() or "démographie" in question_type.lower():
        return prompt_demographie
    elif "économie" in question_type.lower() or "pib" in question_type.lower():
        return prompt_economie
    else:
        return prompt  # Version complète par défaut

# Exemple d'utilisation dans votre fonction generate :
async def generate(state: GraphState, *, config: RagConfiguration):
    """Generate avec prompt amélioré"""
    print("---GENERATE---")
    messages = state.messages
    documents = state.documents

    question = " ".join(msg.content for msg in messages if isinstance(msg, HumanMessage))

    # Sélection du prompt selon le type de question
    selected_prompt = get_ansd_prompt(question)
    
    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    rag_chain = selected_prompt | model

    response = await rag_chain.ainvoke({
        "context": "\n".join(doc.page_content for doc in documents),
        "question": question
    })
    
    return {
        "messages": [response],
        "documents": documents
    }
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
graph.name = "SimpleRag"
