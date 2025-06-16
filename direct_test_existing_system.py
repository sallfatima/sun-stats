# =============================================================================
# TEST DIRECT AVEC VOTRE SYSTÈME EXISTANT
# =============================================================================

# ÉTAPE 1: Vérifiez que votre graph.py contient les corrections
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

# ÉTAPE 2: Test manuel simple
def manual_image_test():
    """Test manuel pour voir si on peut accéder aux images."""
    
    print("\n🧪 TEST MANUEL D'ACCÈS AUX IMAGES")
    print("=" * 40)
    
    from pathlib import Path
    
    # Test d'accès direct aux images
    images_dir = Path('images')
    if images_dir.exists():
        images = list(images_dir.glob('*.png'))
        print(f"✅ Dossier images accessible: {len(images)} images")
        
        # Test avec une image spécifique
        if images:
            test_image = images[0]
            print(f"📸 Test image: {test_image.name}")
            
            # Simuler l'affichage Chainlit
            try:
                # Test si on peut lire l'image
                with open(test_image, 'rb') as f:
                    size = len(f.read())
                print(f"✅ Image lisible: {size} bytes")
                
                # Test du chemin
                print(f"✅ Chemin absolu: {test_image.absolute()}")
                print(f"✅ Chemin relatif: {test_image}")
                
                return str(test_image)
                
            except Exception as e:
                print(f"❌ Erreur lecture image: {e}")
                return None
    else:
        print("❌ Dossier images non accessible")
        return None

# ÉTAPE 3: Test avec l'interface réelle
def test_real_system():
    """Instructions pour tester avec votre système réel."""
    
    print("\n🎯 TEST AVEC VOTRE SYSTÈME RÉEL")
    print("=" * 40)
    
    instructions = """
ÉTAPES POUR TESTER MAINTENANT:

1. 🔧 APPLIQUEZ LES CORRECTIONS dans src/simple_rag/graph.py:

   A) Ajoutez cette fonction après les imports:
   
def debug_visual_elements(visual_elements):
    print(f"\\n🔍 DEBUG VISUAL: {len(visual_elements)} éléments détectés")
    for i, element in enumerate(visual_elements, 1):
        metadata = element['metadata']
        print(f"📊 Élément {i}: {metadata.get('pdf_name', 'N/A')}, page {metadata.get('page_num', 'N/A')}")
        # Afficher métadonnées d'image
        for key, value in metadata.items():
            if 'image' in key.lower() or 'path' in key.lower():
                print(f"   🖼️ {key}: {value}")

   B) Dans process_and_display_visual_elements, ajoutez au début:
   
debug_visual_elements(visual_elements)

   C) Remplacez display_chart_element par:
   
async def display_chart_element(metadata, content):
    print(f"🔍 Métadonnées complètes: {metadata}")
    
    # Recherche intelligente de l'image
    image_path = None
    
    # Méthode 1: Depuis métadonnées
    for key in ['image_path', 'source', 'file_path']:
        if key in metadata and metadata[key]:
            test_path = Path(metadata[key])
            if test_path.exists():
                image_path = str(test_path)
                break
            # Essayer dans le dossier images/
            alt_path = Path('images') / test_path.name
            if alt_path.exists():
                image_path = str(alt_path)
                break
    
    # Méthode 2: Par pattern PDF + page
    if not image_path:
        pdf_name = metadata.get('pdf_name', '')
        page_num = metadata.get('page_num', metadata.get('page', ''))
        
        if pdf_name and page_num:
            import glob
            pattern = f"images/*{Path(pdf_name).stem}*{page_num}*.png"
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
    
    # Affichage
    if image_path and Path(image_path).exists():
        print(f"✅ IMAGE TROUVÉE: {image_path}")
        
        try:
            import chainlit as cl
            await cl.Message(content=f"📊 Graphique: {metadata.get('pdf_name', '')}").send()
            elements = [cl.Image(name="chart", path=str(image_path), display="inline")]
            await cl.Message(content="", elements=elements).send()
            await cl.Message(content=f"✅ Image affichée: {Path(image_path).name}").send()
        except:
            print(f"🖼️ Image disponible: {image_path}")
        
        return True
    else:
        print(f"❌ IMAGE NON TROUVÉE pour {metadata.get('pdf_name', 'N/A')}")
        print(f"   Métadonnées: {metadata}")
        return False

2. 🧪 TESTEZ en posant votre question:
   "Graphique évolution de la population"

3. 🔍 REGARDEZ LES LOGS pour voir:
   - DEBUG VISUAL: X éléments détectés
   - Les métadonnées d'image
   - Les tentatives de recherche
   - ✅ IMAGE TROUVÉE ou ❌ IMAGE NON TROUVÉE

4. 📊 SI IMAGE NON TROUVÉE:
   - Vérifiez les métadonnées affichées
   - Cherchez manuellement l'image correspondante
   - Adaptez le pattern de recherche
"""
    
    print(instructions)

if __name__ == "__main__":
    print("🇸🇳 DIAGNOSTIC SYSTÈME IMAGES ANSD")
    print("=" * 60)
    
    # Vérifier les corrections
    corrections_ok = check_graph_corrections()
    
    # Test manuel
    test_image = manual_image_test()
    
    # Instructions
    test_real_system()
    
    print("\n" + "=" * 60)
    if corrections_ok:
        print("✅ PRÊT À TESTER - Lancez votre système avec une question visuelle")
    else:
        print("🔧 APPLIQUEZ D'ABORD LES CORRECTIONS dans graph.py")
    
    if test_image:
        print(f"📸 Image test disponible: {test_image}")
    
    print("💡 Objectif: Voir '✅ IMAGE TROUVÉE' dans les logs")