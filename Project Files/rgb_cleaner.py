from PIL import Image
import os

def clean_image_folder(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'gif')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        if img.mode in ('P', 'LA', 'RGBA'):
                            img.convert('RGB').save(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

clean_image_folder('output_dataset')