import os
import numpy as np
import json
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model = load_model('healthy_vs_rotten.h5')

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

BASE_UPLOAD_FOLDER = 'static/uploads'
FRESH_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'fresh')
ROTTEN_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'rotten')

os.makedirs(FRESH_FOLDER, exist_ok=True)
os.makedirs(ROTTEN_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/inspect', methods=['GET', 'POST'])
def inspect():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            original_filename = image_file.filename
            temp_path = os.path.join(BASE_UPLOAD_FOLDER, original_filename)
            image_file.save(temp_path)

            # Load and preprocess image
            img = load_img(temp_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_label = index_to_class[predicted_index]
            confidence = round(100 * np.max(prediction), 2)

            # Decide target folder
            if 'fresh' in predicted_label.lower() or 'healthy' in predicted_label.lower():
                target_folder = FRESH_FOLDER
            else:
                target_folder = ROTTEN_FOLDER

            final_path = os.path.join(target_folder, original_filename)
            os.replace(temp_path, final_path)  # Move file to correct folder

            # For HTML display, construct relative path
            image_path = '/' + final_path.replace("\\", "/")

            return render_template('inspect.html',
                                   prediction=predicted_label,
                                   confidence=confidence,
                                   image_path=image_path)

    return render_template('inspect.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
