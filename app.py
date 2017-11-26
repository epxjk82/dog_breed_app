from flask import Flask, render_template, request, send_from_directory
import os
import urllib
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import xception
# from keras.applications import resnet50
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import cv2
import numpy as np
from glob import glob
from load import *
import csv

app = Flask(__name__)

#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

PORT = '5000'
UPLOAD = 'uploads'

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/download', methods=['POST'])
def download_image():
    address = request.form['address']
    filename = os.path.join(UPLOAD, address.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(urllib.urlopen(address).read())
    return predict(filename)


@app.route('/upload', methods=['POST'])
def upload_image():
    f = request.files['image']
    filename = os.path.join(UPLOAD, f.filename)
    f.save(filename)
    return predict(filename)


@app.route('/' + UPLOAD + '/<path:path>')
def serve_files(path):
    return send_from_directory(UPLOAD, path)


def predict(filename):

    with open('static/dog_names.csv', 'r') as f:
        reader = csv.reader(f)
        dog_names = list(reader)[0]
    # dog_names = [item[32:-1] for item in sorted(glob("C:/datasets/dogImages/train/*/"))]
    # print (dog_names)
    img = image.load_img(filename, target_size=(299,299))
    x = image.img_to_array(img)
    input_tensor = np.expand_dims(x,axis=0)
    input_tensor = input_tensor/255.
    print ("input_tensor shape:",input_tensor.shape)
    with graph.as_default():
    	#perform the prediction
        pred = model.predict(input_tensor)
        result = []
        result.append(dog_names[np.argmax(pred)])
        pred_breed = dog_names[np.argmax(pred)]
        pred_breed = pred_breed.replace('_', ' ')
        # decode_result = decode_predictions(pred)[0]

        # for r in decode_result:
        #     result.append({'name':r[1], 'prob':r[2]*100})
        return render_template('predict.html',
                               filename=filename,
                               predictions=pred_breed)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # app.run(host='0.0.0.0', port=PORT)
    # app.run(debug=True)
    app.run()
