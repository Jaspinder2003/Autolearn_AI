import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
MODEL_PATH = "./models/cnn_model(1).h5"
model = tf.keras.models.load_model(MODEL_PATH)

classes = ['car', 'plane']

# Define image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Predict function
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction)
    print(f"Predicted Class: {classes[predicted_class]}| confidence: {confidence}")

    print("--------------------")
    

# Run script from command line with an image path
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_runner.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    predict_image(image_path)
