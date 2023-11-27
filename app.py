import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
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
    st.image(image_input, caption="Uploaded Image.", use_column_width=True)
    image = Image.open(image_input)
    if st.button("Predict"):
        if image_input is not None:
            model=load_model("models/cnn_model.h5")
            img = image.resize((128, 128))
            img_array = np.array(img)
            img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                st.write("Tumor Detected",res)
            else:
                st.write("No Tumor",res)
    else:
         st.warning("Upload an image.")

elif selected_option == "Sentiment Classification":
    model_choice = st.selectbox("Select Model", ["Perceptron", "Backpropagation","DNN","RNN","LSTM"])
    if model_choice == "Perceptron":
        st.subheader("Movie review classification using perceptron")
        text_input = st.text_area("Enter a movie review" )
        if st.button('Predict'):
            if text_input:
                with open('models/imdb_perceptron_model.pkl', 'rb') as file:
                    model = pickle.load(file)
                num_words=1000
                max_len=200
                word_index = imdb.get_word_index()
                input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in text_input.split()]
                padded_sequence = pad_sequences([input_sequence], maxlen=max_len)
                prediction = model.predict(padded_sequence)[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                st.write(f"{sentiment}")
            else:
                 st.warning("Enter a movie review first.")

    elif model_choice == "Backpropagation":
            st.subheader("Movie review classification using Backpropagation")
            text_input = st.text_area("Enter a movie review" )
            if st.button('Predict'):
                if text_input:
                    with open('models/backprop_model.pkl', 'rb') as file:
                        model = pickle.load(file)
                    num_words=1000
                    max_len=200
                    word_index = imdb.get_word_index()
                    input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in text_input.split()]
                    padded_sequence = pad_sequences([input_sequence], maxlen=max_len)
                    prediction = model.predict(padded_sequence)[0]
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    st.write(f"{sentiment}")
                else:
                     st.warning("Enter a movie review first.")

    elif model_choice == "DNN":
                st.subheader("SMS Spam/ham classification using DNN")
                text_input = st.text_area("Enter an sms text")
                if st.button("Predict"):
                    model = load_model("models/dnn_model.h5")
                    with open('tokeniser.pkl', 'rb') as file:
                        tokeniser = pickle.load(file)
                    if text_input:
                        sequence = tokeniser.texts_to_sequences([text_input])
                        padded_sequence = pad_sequences(sequence, maxlen=10)
                        prediction = model.predict(padded_sequence)[0][0]
                        if prediction >= 0.5:
                            st.success("Ham")
                        else:
                            st.write("Spam")
                    else:
                        st.warning("Please enter an sms text first.")

    elif model_choice == "RNN":
            
            st.subheader("SMS Spam/ham classification using RNN")
            text_input = st.text_area("Enter an sms text")
            if st.button("Predict"):
                model = load_model("models/rnn_model.h5")
                with open('tokeniser.pkl', 'rb') as file:
                    tokeniser = pickle.load(file)
                if text_input:
                    encoded_text = tokeniser.texts_to_sequences([text_input])
                    padded_text = tf.keras.preprocessing.sequence.pad_sequences(encoded_text, maxlen=10, padding='post')
                    prediction = model.predict(padded_text)
                    if prediction >= 0.5:
                        if prediction >= 0.5:
                            st.success("Ham")
                        else:
                            st.write("Spam")
                else:
                    st.warning("Please enter an sms text first.")
            
    elif model_choice == "LSTM":
            st.subheader("Movie review classification using LSTM")
            model = load_model("models/lstm_model.h5")
            text_input = st.text_area("Enter a movie review", "")
            if st.button("Predict"):
                if text_input:
                    tokenizer = Tokenizer(num_words=5000)
                    input_sequence = tokenizer.texts_to_sequences([text_input])
                    input_padded = pad_sequences(input_sequence, maxlen=100)
                    prediction = model.predict(input_padded)
                    if prediction[0][0]<0.5 :
                        st.write("Negative")
                    else:
                        st.write("Positive")
                else:
                     st.warning("Enter a movie review first.")




