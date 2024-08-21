from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import cv2
import os
from werkzeug.utils import secure_filename
from src.pipeline.predict_pipeline import PredictImage

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        file=request.files['file']
        filename = secure_filename(file.filename)
        FOLDER_PATH = os.path.join("static")
        os.makedirs(FOLDER_PATH, exist_ok=True)
        FILE_PATH = os.path.join("static", filename)
        file.save(FILE_PATH)
        predict_obj = PredictImage(FILE_PATH)
        prediction = predict_obj.predict()
        return render_template('complete.html', image=FILE_PATH, imageresults=prediction)

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)    