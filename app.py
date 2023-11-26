import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess image for tumor detection
def preprocess_image(image_path, target_size=(180, 180)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)
    return img

# Main content
st.title("Deep Learning Algorithms")

selected_option = st.radio("Choose an option", ["Tumor Detection", "Sentiment Classification"])

# Upload image if "Tumor Detection" is selected
if selected_option == "Tumor Detection":
    st.title("Tumor Detection using CNN")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True,width=10)

    # Add a "Predict" button
    if st.button("Predict"):

        # Check if an image is uploaded before attempting to process it
        if uploaded_image is not None:
            model = load_model("models/cnn_model.h5")
            # Preprocess the image for tumor detection
            processed_image = preprocess_image(uploaded_image)
            # Make the prediction
            result = model.predict(processed_image)
            # Display the result
            if result[0][0] > 0.5:  # Assuming binary classification
                st.write("Tumor Detected")
            else:
                st.write("No Tumor")
        else:
            st.warning("Please upload an image before clicking 'Predict'")