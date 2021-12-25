import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

def pred_stats(prediction_probabilities):
  top_5_pred_indexes = prediction_probabilities.argsort()[-5:][::-1]
  top_5_pred_labels = unique_breeds[top_5_pred_indexes]
  top_5_pred_values = prediction_probabilities[top_5_pred_indexes]
  fig, ax = plt.subplots()
  set_color = ['gray' if (x < tf.reduce_max(top_5_pred_values)) else 'green' for x in top_5_pred_values ]
  bar_plot = ax.bar(np.arange(len(top_5_pred_labels)), top_5_pred_values, color=set_color)
  for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                str(f"{top_5_pred_values[idx]*100:.1f}%"),
                ha='center', va='bottom', rotation=0, c="red")
  ax.set_xticks(np.arange(len(top_5_pred_labels)), labels=top_5_pred_labels, rotation="vertical")
  st.pyplot(fig)
  