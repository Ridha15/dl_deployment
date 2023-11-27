import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from Perceptron import  Perceptron

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

    preprocessed_image = preprocess_input(img_array)
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

elif selected_option == "Sentiment Classification":
    
    model_choice = st.selectbox("Select Model", ["Perceptron", "Backpropagation","DNN","RNN","LSTM","GRU"])
    if model_choice == "Perceptron":
        st.subheader("Movie review classification")
        text_input = st.text_area("Enter a movie review" )
        if st.button('Predict'):
            with open('models/imdb_perceptron_model.pkl', 'rb') as file:
                model = pickle.load(file)
            num_words=1000
            max_len=200
            word_index = imdb.get_word_index()
            input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in text_input.split()]
            padded_sequence = pad_sequences([input_sequence], maxlen=max_len)
            prediction = model.predict(padded_sequence)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            st.write(f"Predicted Sentiment: {sentiment}")
    elif model_choice == "Backpropagation":
        st.subheader("Movie review classification")
        text_input = st.text_area("Enter a movie review" )
        if st.button('Predict'):
            with open('models/backprop_model.pkl', 'rb') as file:
                model = pickle.load(file)
            num_words=1000
            max_len=200
            word_index = imdb.get_word_index()
            input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in text_input.split()]
            padded_sequence = pad_sequences([input_sequence], maxlen=max_len)
            prediction = model.predict(padded_sequence)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            st.write(f"Predicted Sentiment: {sentiment}")

    elif model_choice == "DNN":
            st.subheader("SMS Spam/ham classification")
            text_input = st.text_area("Enter an sms text")
            if st.button("Predict"):
                model = load_model("models/dnn_model/h5")
                with open('tokeniser.pkl', 'rb') as file:
                    tokeniser = pickle.load(file)
                if text_input:
                    sequence = tokeniser.texts_to_sequences([text_input])
                    padded_sequence = pad_sequences(sequence, maxlen=10)
                    prediction = model.predict(padded_sequence)[0][0]
                    if prediction >= 0.5:
                        st.success("not spam")
                    else:
                        st.write("Spam")
                else:
                    st.write("Please enter an sms text first.")


