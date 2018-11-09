import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import requests
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64

img_width, img_height = 224, 224

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file): 
    x_img = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x_img)
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
        array = model.predict(x)
 
    result = array[0][0]
    #answer = np.argmax(result)
    if result <= 0.5:
        print("Label: Basic")
        label='Basic '+' ,probability='+ str(round(result,2))
    else:
        print("Label: Premium")
        label='Premium '+' ,probability='+ str(round(result,2))
    return label

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('VGG16-transferlearning_2.model')
    graph = tf.get_default_graph()

@app.route("/")
def template():
    return render_template('template.html', label='Premium, probability=0.99', imagesource='./uploads/template.png')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            label = predict(file_path)
            filename = my_random_string(6) + filename
            
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
    init()
    app.run()
            