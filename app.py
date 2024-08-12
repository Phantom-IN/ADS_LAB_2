from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mnist_model.h5')
# Load the pre-trained spam detection model
with open('spam_detection_model.pkl', 'rb') as f:
    spam_model = pickle.load(f)

def prepare_image(image):
    # Convert the image to grayscale and resize it to 28x28
    image = image.convert('L')
    image = image.resize((28, 28))
    image = img_to_array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255.0
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/mnist', methods=['GET', 'POST'])
def mnist():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('mnist.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('mnist.html', error="No selected file")

        # Read the image and prepare it for prediction
        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image)
        
        # Make prediction
        predictions = model.predict(prepared_image)
        digit = np.argmax(predictions[0])

        return render_template('mnist.html', digit=digit)

    return render_template('mnist.html')

@app.route('/spam', methods=['GET', 'POST'])
def spam():
    if request.method == 'POST':
        message = request.form['message']
        prediction = spam_model.predict([message])[0]
        return render_template('spam.html', message=message, prediction=prediction)

    return render_template('spam.html')

if __name__ == '__main__':
    app.run(debug=True)
