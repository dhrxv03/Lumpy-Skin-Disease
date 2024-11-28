import os
# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr

#suppress tensorflow errores
tf.get_logger().setLevel('ERROR')

# Load the trained models
densenet_model = load_model('./trained_models/densenet_lumpy_skin.h5')
behavioral_model = load_model('./trained_models/behavioral_lsd_model.h5')

# Define input size
input_size = (224, 224)


# Load and preprocess the image for DenseNet prediction
def preprocess_image(image):
    img = image.resize(input_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Prediction function using DenseNet
def predict_with_densenet(image):
    img_array = preprocess_image(image)
    prediction = densenet_model.predict(img_array)[0][0] * 100
    result = "Lumpy Skin Disease" if prediction > 50 else "Healthy"
    confidence = f"{prediction:.2f}%"

    return f"{result} {confidence}"


#gradio
def gradio_interface(image, appetite, movement, grazing, social, drinking, breathing, posture, skin, milk, temp):
    result = predict_with_densenet(image)
    return result


#behavorial dataset features
behavioral_options = {
    "Appetite": ["normal", "reduced"],
    "Movement": ["active", "lethargic"],
    "Grazing": ["normal", "reduced"],
    "Social": ["interacts", "isolation"],
    "Drinking": ["regular", "reduced"],
    "Breathing": ["normal", "labored"],
    "Posture": ["normal", "difficulty"],
    "Skin": ["smooth", "lesions"],
    "Milk": ["normal", "decreased"],
    "Temp": ["normal", "fever"],
}

#build gradio interface
inputs = [
    gr.Image(type="pil", label="Upload an Image of a Cow"),
    *[gr.Radio(label=key, choices=value) for key, value in behavioral_options.items()]
]

interface = gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=gr.Textbox(lines=5, max_lines=10, label="Prediction Result"),
    title="Lumpy Skin Disease Detection",
    description="Upload an image of a cow to predict whether it has Lumpy Skin Disease. "
)

if __name__ == "__main__":
    interface.launch()
