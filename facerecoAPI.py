import cv2
from facereco import btp1
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import time

app=Flask(__name__)

@app.route('/', methods=['POST'])
def get_img():
    
    file=request.files['image']
    filename = file.filename
    file.save( filename)
    image=cv2.imread(filename)
    imgrecog = btp1(image)
    os.remove(filename)
    return str(imgrecog)

if __name__=='__main__':
    app.run()