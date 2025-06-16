# diagnostic_images.py
import asyncio
from pathlib import Path

async def quick_diagnosis():
    # Vérifier les dossiers d'images
    image_dirs = ['images', 'data/images', './images', 'static/images']
    
    for dir_path in image_dirs:
        if Path(dir_path).exists():
            images = list(Path(dir_path).glob('*.png')) + list(Path(dir_path).glob('*.jpg'))
            print(f"✅ {dir_path}: {len(images)} images")
            if images:
                print(f"   Exemple: {images[0]}")
        else:
            print(f"❌ {dir_path}: n'existe pas")

asyncio.run(quick_diagnosis())