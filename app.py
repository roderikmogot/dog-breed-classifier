import streamlit as st
from helper import get_pred_label, load_model, load_img, convert, pred_stats, unique_breeds
model = load_model("big_dog_model.h5")

st.markdown("# Dog Breed Classifier")
sd = st.sidebar.selectbox(
  "Types of dog that can be classified",
  unique_breeds
)

image_file = st.file_uploader("Choose an image file", type=['jpeg', 'jpg', 'png'])

if image_file:
  st.image(load_img(image_file))

  img = convert(image_file)

  custom_preds = model.predict(img)
  custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]

  st.write(f"Predicted: {custom_pred_labels[0]}, with an accuracy of {custom_preds[0].max()*100:.0f}%!")

  st.markdown("<h3 style='text-align: left; color: white;'>Top 5 predictions</h3>", unsafe_allow_html=True)

  pred_stats(custom_preds[0])
