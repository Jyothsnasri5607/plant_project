import numpy as np
import cv2
from tensorflow.keras.models import load_model
from disease_info import disease_solutions

#Load Model
model = load_model("plant_model.h5")

#update class names Exactly as printed earlier
class_names = ['early_blight', ' healthy', 'late_blight']

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)

    pred = model.predict(img)
    return class_names[np.argmax(pred)]

predicted class = class_names[np.argmax(preds)]

info = disease_solutions.get(predicted_class, None)

if info:
    print("\n 🌿 Disease:", predicted_class)
    print("📖 Description:", info["description"])

    print("\n💊 Treatment:")
    for step in info["treatment"]:
        print("-", step)
else:
    print("No info available for this disease.")

result = predict("test.jpg")
print("Prediction:", result)