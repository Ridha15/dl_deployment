import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess image for tumor detection
def preprocess_image(image):
    image = image.resize((299, 299))
    image_array = np.array(image)
    preprocessed_image = preprocess_input(image_array)

    return preprocessed_image

def make_prediction_cnn(image, image_model):
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

    preprocessed_image = preprocess_image(img_array)
    prediction = image_model.predict(preprocessed_image)

    if prediction > 0.5:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")

# Main content
st.title("Deep Learning Algorithms")

selected_option = st.radio("Choose an option", ["Tumor Detection", "Sentiment Classification"])

# Upload image if "Tumor Detection" is selected
if selected_option == "Tumor Detection":
    st.title("Tumor Detection using CNN")

    st.subheader("Image Input")
    image_input = st.file_uploader("Choose an image...", type="jpg")

    if image_input is not None:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if st.button("Predict"):
            make_prediction_cnn(image, image_model)