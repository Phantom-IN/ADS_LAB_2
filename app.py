from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mnist_model.h5')

# Load the pre-trained model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    spam_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/spam', methods=['GET', 'POST'])
def spam():
    if request.method == 'POST':
        message = request.form['message']
        
        # Preprocess the input message
        ps = PorterStemmer()
        # Applying Regular Expression
        msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', message)
        msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
        msg = re.sub('£|\$', 'moneysymb', msg)
        msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', msg)
        msg = re.sub('\d+(\.\d+)?', 'numbr', msg)
        msg = re.sub('[^\w\d\s]', ' ', msg)
        msg = msg.lower()
        msg = msg.split()
        msg = [ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
        msg = ' '.join(msg)
        
        # Transform the input message
        msg_vector = vectorizer.transform([msg]).toarray()
        
        # Predict using the model
        prediction = spam_model.predict(msg_vector)[0]
        prediction = 'Spam' if prediction == 1 else 'Ham'
        
        return render_template('spam.html', message=message, prediction=prediction)

    return render_template('spam.html')

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


if __name__ == '__main__':
    app.run(debug=True)
