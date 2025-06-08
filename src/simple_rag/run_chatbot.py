# src/simple_rag/run_chatbot.py

from src.simple_rag.graph import RAGGraph

def main():
    chatbot = RAGGraph()
    chat_history = []

    print("=== Chatbot RGPH – ANSD (avec FAISS local) (tapez 'exit' pour quitter) ===\n")
    while True:
        question = input("Vous : ")
        if question.strip().lower() in ["exit", "quit", "q"]:
            print("Au revoir ! 👋")
            break

        answer, sources = chatbot.ask(question, chat_history)
        print("\n[Bot] :", answer, "\n")
        print("Documents utilisés pour la réponse :")
        for doc in sources:
            print(f" • {doc.metadata.get('source_pdf', 'inconnu')} (chunk id {doc.metadata.get('chunk', '?')})")
        print("\n" + "-" * 60 + "\n")
        chat_history.append((question, answer))

if __name__ == "__main__":
    main()
