import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

model = tf.keras.models.load_model("classification_model.keras")

def classify(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = preprocess_input(img, data_format=None)
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    
    return model.predict(img)[0][0] > 0.5