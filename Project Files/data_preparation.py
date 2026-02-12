# data_preparation.py
import os, shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(input_dir='data/Fruit and Vegetable Diseases Dataset', output_dir='output_dataset', limit_per_class=200):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for cls in os.listdir(input_dir):
        src = os.path.join(input_dir, cls)
        if not os.path.isdir(src): continue

        imgs = os.listdir(src)[:limit_per_class]
        trainval, test = train_test_split(imgs, test_size=0.2, random_state=42)
        train, val = train_test_split(trainval, test_size=0.25, random_state=42)

        for split, names in [('train', train), ('val', val), ('test', test)]:
            dst = os.path.join(output_dir, split, cls)
            os.makedirs(dst, exist_ok=True)
            for img in names:
                shutil.copy(os.path.join(src, img), os.path.join(dst, img))
        print(f"{cls}: train {len(train)}, val {len(val)}, test {len(test)}")

if __name__ == "__main__":
    prepare_dataset()
