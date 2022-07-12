from flask import Flask,render_template,request
from PIL import Image
import numpy as np
import cv2
import os
import base64
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import pickle

image_labels = pickle.load(open("trash_label.pkl","rb"))
model = pickle.load(open("model (1).pkl","rb"))
app = Flask(__name__)

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/submit/',methods=['POST'])
def submit():
    file = request.files['image']
    file.save(os.path.join("UPLOAD", "file.jpg"))
    s=predict_disease("UPLOAD/file.jpg")
    os.remove("UPLOAD/file.jpg")
    return s

DEFAULT_IMAGE_SIZE = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    result = model.predict_classes(np_image)
    return (image_labels.classes_[result][0])


if __name__ == '__main__':
   app.run()