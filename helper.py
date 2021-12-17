import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

labels_csv = pd.read_csv("labels.csv")

labels = np.array(labels_csv['breed'])
unique_breeds = np.unique(labels)

@st.cache(allow_output_mutation=True)
def load_model(model_path):
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
  return model

@st.cache(allow_output_mutation=True)
def load_img(img_path):
  img = Image.open(img_path)
  return img

def convert(image_file):
  img_array = np.array(load_img(image_file))
  img_array = img_array / 255.0
  img = tf.image.resize(img_array, size=(224,224))
  img = tf.expand_dims(img, axis=0)
  return img

def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]