from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load mô hình đã lưu
model = tf.keras.models.load_model('handwriting_recognition_model.h5')

def preprocess_image(image):
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array
    image = np.array(image)
    # Normalize pixel values
    image = image / 255.0
    # Flatten the image to match the expected input shape
    image = image.flatten()
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@cross_origin()
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(img)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
