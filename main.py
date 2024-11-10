import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from gtts import gTTS
import os
import platform
import threading

# Load the pre-trained model
model = tf.keras.models.load_model('cashRec.h5')

# Class labels for the Nepali notes
classes = ['fifty', 'five', 'fivehundred', 'hundred', 'ten', 'thousand', 'twenty']

# Function to classify the image
def classify_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    y_pred = classes[index]
    return y_pred

# Function to speak the predicted result using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")

    # Play the sound based on the platform (macOS)
    if platform.system() == "Darwin":  # macOS
        os.system("afplay output.mp3")

# Streamlit app setup
st.title("Nepali Cash Recognition")
uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the image and display the result
    prediction = classify_image(uploaded_file)
    output = f"The Note is {prediction} rupees."
    st.write(output)

    # Run the speech in a separate thread
    threading.Thread(target=speak, args=(f"{prediction} Rupees",)).start()
