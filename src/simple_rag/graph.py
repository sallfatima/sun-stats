# src/simple_rag/graph.py

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from .configuration import RAGConfig

class RAGGraph:
    def __init__(self):
        self.config = RAGConfig()

        # Charger l’index FAISS local (avec autorisation de désérialisation)
        self.vectorstore = FAISS.load_local(
            self.config.faiss_index_path,
            self.config.embedding_model,
            allow_dangerous_deserialization=True
        )

        # LLM ChatOpenAI
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.config.openai_api_key,
            temperature=0.0
        )

        # Chaîne RAG conversationnelle
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

    def ask(self, question: str, chat_history: list = None):
        if chat_history is None:
            chat_history = []
        result = self.chain({"question": question, "chat_history": chat_history})
        return result["answer"], result["source_documents"]
