# generate_class_indices.py

import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_class_indices(dataset_path='output_dataset/train', output_file='class_indices.json'):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical'
    )

    class_indices = generator.class_indices
    with open(output_file, 'w') as f:
        json.dump(class_indices, f)

    print(f"Class indices saved to: {output_file}")
    print("Class index mapping:")
    print(json.dumps(class_indices, indent=2))

if __name__ == "__main__":
    generate_class_indices()
