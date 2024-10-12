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

# Set TensorFlow logger to only show errors
tf.get_logger().setLevel('ERROR')

# Define paths
image_path = 'sample.jpg'
logs_dir = './Logs'
model_path = './trained_models/densenet_lumpy_skin.h5'
input_size = (224, 224)

# Function to ensure directory exists
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)  # Simplified directory creation

# Ensure Logs directory exists
ensure_dir(logs_dir)

# Load DenseNet model
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist")
model = load_model(model_path)

# Predict function using DenseNet
def predict_with_densenet(image_path, model):
    # Load and preprocess the image
    img = load_img(image_path, target_size=input_size)
    img_array = img_to_array(img) / 255.0  # Rescale the image directly
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for the model

    # Make prediction
    return model.predict(img_array)[0][0] * 100

# Get prediction
prediction = predict_with_densenet(image_path, model)

# Display prediction
result = "Lumpy Skin Disease" if prediction > 50 else "Healthy"
print(f"DenseNet Prediction: {result}: {prediction:.2f}%")

# Create bar graph
plt.figure(figsize=(6, 6))
plt.bar(["DenseNet"], [prediction], color='blue')
plt.xlabel('Model Name')
plt.ylabel('Prediction Percentage')
plt.title('DenseNet Prediction for Lumpy Skin Disease')
plt.ylim([0, 110])
plt.yticks(np.arange(0, 101, 10))
plt.text("DenseNet", prediction + 1, f'{prediction:.2f}%', ha='center', va='bottom')

# Save the results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
png_path = os.path.join(logs_dir, f'log_{timestamp}.png')
csv_path = os.path.join(logs_dir, f'log_{timestamp}.csv')

plt.savefig(png_path)
plt.close()

# Save prediction results to CSV
df = pd.DataFrame([["DenseNet", prediction]], columns=['Model Name', 'Prediction Percentage'])
df.to_csv(csv_path, index=False)
print(f"Bar graph saved to {png_path}")
print(f"CSV file saved to {csv_path}")
