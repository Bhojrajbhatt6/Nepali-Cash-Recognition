import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import pyttsx3
import threading
import time

# Initialize pyttsx3 engine
engine = pyttsx3.init()

model = tf.keras.models.load_model('cashRec.h5')

classes = ['fifty', 'five', 'fivehundred', 'hundred', 'ten', 'thousand', 'twenty']

# Lock to prevent concurrent speech engine usage
engine_lock = threading.Lock()

def classify_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    y_pred = classes[index]
    return y_pred

# Function to speak the prediction result
def speak(prediction):
    with engine_lock:  # Ensure that only one thread can use the engine at a time
        engine.say(f"{prediction} Rupees")
        engine.setProperty('volume', 1)
        engine.runAndWait()

# Function to handle speech synthesis in a separate thread
def thread_speak(prediction):
    try:
        speech_thread = threading.Thread(target=speak, args=(prediction,))
        speech_thread.start()
    except Exception as e:
        st.error(f"Error occurred while trying to speak: {e}")

uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = classify_image(uploaded_file)
    output = f"The Note is {prediction} rupees."
    st.write(output)
    
    # Ensure enough time for speech synthesis to finish before returning control to Streamlit
    thread_speak(prediction)
    time.sleep(1)  # Optional: Ensure speech thread runs before ending the function