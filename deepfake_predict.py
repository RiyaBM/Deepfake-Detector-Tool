from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the model from HDF5 file
loaded_model = load_model('deepfake_image_model.h5')

# Constants
IMG_SIZE = (224, 224)  # Resize images to this size

# Prediction function
def predict_image(model, img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        prediction = model.predict([img_array, img_array])[0][0]  # Pass two identical inputs
        return "real" if prediction > 0.5 else "fake"
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return "Error"

# Example usage
example_images = [
    r"C:\Users\eliza\Downloads\test_gen1.png",
    r"C:\Users\eliza\Downloads\test_gen2.png",
    r"C:\Users\eliza\Downloads\test_real1.png",
    r"C:\Users\eliza\Downloads\test_real2.png",
    r"C:\Users\eliza\Downloads\real_4.png"
]

for img_path in example_images:
    result = predict_image(loaded_model, img_path)
    print(f"Prediction for {img_path}: {result}")
