import json, numpy as np, random, os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_labels(cm='class_indices.json'):
    ci = json.load(open(cm))
    labels = [None]*len(ci)
    for k,v in ci.items(): labels[v] = k
    return labels

def predict_image(img_path, model_path='healthy_vs_rotten.h5'):
    model = load_model(model_path)
    labels = load_labels()
    img = load_img(img_path, target_size=(224,224))
    arr = preprocess_input(img_to_array(img)[None,...])
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    print(img_path, "→", labels[idx], f"({preds[idx]*100:.2f}%)")

def random_test(dir='output_dataset/val'):
    cls = random.choice(os.listdir(dir))
    fname = random.choice(os.listdir(os.path.join(dir,cls)))
    predict_image(os.path.join(dir,cls,fname))

if __name__ == "__main__":
    random_test()
