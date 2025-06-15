# Créez un fichier debug_pinecone.py pour diagnostiquer votre index

import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def diagnose_pinecone_index():
    """
    Diagnostic de l'index Pinecone pour vérifier les données 2023
    """
    try:
        print("🔌 Diagnostic de l'index Pinecone...")
        
        # Configuration
        embeddings = OpenAIEmbeddings()
        index_name = os.getenv('PINECONE_INDEX', 'index-ansd')
        
        # Connexion à Pinecone
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        print(f"✅ Connexion réussie à l'index: {index_name}")
        
        # Test 1: Recherche spécifique RGPH-5 2023
        queries_test = [
            "RGPH-5 2023",
            "population Sénégal 2023",
            "17 731 714 habitants",
            "recensement 2023",
            "RGPH 2023"
        ]
        
        for query in queries_test:
            print(f"\n🔍 Test requête: '{query}'")
            
            docs = vectorstore.similarity_search(query, k=5)
            print(f"📄 {len(docs)} documents trouvés")
            
            for i, doc in enumerate(docs, 1):
                metadata = getattr(doc, 'metadata', {})
                content_preview = doc.page_content[:100].replace('\n', ' ')
                
                print(f"  {i}. PDF: {metadata.get('pdf_name', 'Unknown')}")
                print(f"     Page: {metadata.get('page', 'N/A')}")
                print(f"     Contenu: {content_preview}...")
                
                # Vérifier si contient 2023
                has_2023 = '2023' in doc.page_content.lower()
                has_rgph5 = 'rgph-5' in doc.page_content.lower() or 'rgph5' in doc.page_content.lower()
                has_population_2023 = '17 731 714' in doc.page_content or '17731714' in doc.page_content
                
                print(f"     🏷️ 2023: {'✅' if has_2023 else '❌'} | RGPH-5: {'✅' if has_rgph5 else '❌'} | Pop 2023: {'✅' if has_population_2023 else '❌'}")
        
        # Test 2: Recherche par métadonnées
        print(f"\n🔍 Recherche documents avec '2023' dans le nom...")
        docs_2023 = vectorstore.similarity_search("RGPH 2023 population", k=20)
        
        docs_with_2023 = []
        for doc in docs_2023:
            metadata = getattr(doc, 'metadata', {})
            pdf_name = metadata.get('pdf_name', '')
            content = doc.page_content.lower()
            
            if '2023' in pdf_name.lower() or '2023' in content:
                docs_with_2023.append(doc)
        
        print(f"📊 Documents contenant '2023': {len(docs_with_2023)} sur {len(docs_2023)}")
        
        for doc in docs_with_2023[:3]:  # Afficher les 3 premiers
            metadata = getattr(doc, 'metadata', {})
            print(f"  📄 {metadata.get('pdf_name', 'Unknown')}")
            print(f"     Page: {metadata.get('page', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur diagnostic Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    diagnose_pinecone_index()