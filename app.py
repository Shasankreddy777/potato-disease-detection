from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
model = load_model("model/potato_model.h5")

# Class labels (same order as training)
classes = ['Early Blight', 'Healthy', 'Late Blight']

# Disease solutions
solutions = {
    "Early Blight": "Use fungicides like Mancozeb. Remove infected leaves. Maintain proper spacing.",
    "Late Blight": "Apply copper-based fungicides. Avoid overhead irrigation. Improve air circulation.",
    "Healthy": "Plant is healthy. Continue proper watering and fertilization."
}

# -------------------------------
# Home route (clean UI)
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Save uploaded file
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template(
        'index.html',
        prediction=result,
        solution=solutions[result],
        image_path=filepath
    )

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)