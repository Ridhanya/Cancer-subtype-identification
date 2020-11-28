from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import math


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import openslide as op
from openslide import OpenSlideError
import PIL
from PIL import Image, ImageDraw, ImageFont



app = Flask(__name__)


MODEL_PATH = 'model/New_Model.h5'


model = load_model(MODEL_PATH)
#model._make_predict_function()   


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
	SCALE_FACTOR = 32
	slide = op.open_slide(img_path)
	large_w, large_h = slide.dimensions
	new_w = math.floor(large_w / SCALE_FACTOR)
	new_h = math.floor(large_h / SCALE_FACTOR)
	level = slide.get_best_level_for_downsample(SCALE_FACTOR)
	whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
	whole_slide_image = whole_slide_image.convert("RGB")
	img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	preds = model.predict(x)
	return preds
    



    

    

    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        preds = model_predict(file_path, model)

                 
        #pred_class = decode_predictions(preds, top=1)   
                     
        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)

