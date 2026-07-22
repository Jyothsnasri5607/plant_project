import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from model.disease_info import disease_solutions
from model.gradcam import generate_gradcam
from model.lime_explainer import generate_lime

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model/plant_model.h5")

# -------------------------------
# CLASS NAMES
# -------------------------------
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Healthy"
]

# -------------------------------
# MULTI-LANGUAGE SUPPORT
# -------------------------------
translations = {
    "en": {
        "title": "Plant Disease Detection System",
        "upload": "Upload an image",
        "prediction": "Prediction Result",
        "severity": "Severity Level",
        "recommend": "Smart Recommendations",
        "history": "Prediction History"
    },
    "hi": {
        "title": "पौधा रोग पहचान प्रणाली",
        "upload": "छवि अपलोड करें",
        "prediction": "पूर्वानुमान परिणाम",
        "severity": "गंभीरता स्तर",
        "recommend": "स्मार्ट सुझाव",
        "history": "भविष्यवाणी इतिहास"
    }
}

lang = st.selectbox("🌐 Language", ["en", "hi"])
t = translations[lang]

# -------------------------------
# SESSION HISTORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# SEVERITY FUNCTION
# -------------------------------
def get_severity(confidence):
    if confidence < 0.60:
        return "Low 🟢"
    elif confidence < 0.85:
        return "Medium 🟡"
    else:
        return "High 🔴"

# -------------------------------
# SMART RECOMMENDATION
# -------------------------------
def smart_recommendation(disease, severity):

    if disease == "Healthy":
        return [
            "Maintain current care",
            "Regular watering",
            "Ensure proper sunlight"
        ]

    if severity == "Low 🟢":
        return [
            "Monitor plant daily",
            "Remove affected leaves",
            "Avoid overwatering"
        ]

    elif severity == "Medium 🟡":
        return [
            "Apply fungicide spray",
            "Isolate the plant",
            "Improve air circulation"
        ]

    else:
        return [
            "Remove infected plant immediately",
            "Disinfect gardening tools",
            "Apply strong fungicide treatment"
        ]

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title(f"🌿 {t['title']}")
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(f"📤 {t['upload']}", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    img_cv = np.array(image)

    # PREPROCESS
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICTION
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # SEVERITY
    severity = get_severity(confidence)

    # STORE HISTORY
    st.session_state.history.append({
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "severity": severity
    })

    # GET INFO
    info = disease_solutions.get(predicted_class, None)

    # -------------------------------
    # RESULT UI (MOBILE FRIENDLY)
    # -------------------------------
    st.markdown("---")
    st.subheader(f"🧠 {t['prediction']}")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🌿 {predicted_class}")

    with col2:
        st.write(f"📊 {confidence:.2f}")
        st.write(f"⚠ {severity}")

    # DESCRIPTION
    if info:
        st.markdown("---")

        st.subheader("📖 Description")
        st.info(info["description"])

        st.subheader("💊 Treatment")
        for step in info["treatment"]:
            st.markdown(f"- {step}")
    else:
        st.warning("No information available.")

    # SMART RECOMMENDATIONS
    st.markdown("---")
    st.subheader(f"🤖 {t['recommend']}")

    recommendations = smart_recommendation(predicted_class, severity)

    for rec in recommendations:
        st.markdown(f"- {rec}")

    # GRADCAM
    st.markdown("---")
    st.subheader("🔥 Grad-CAM")

    try:
        gradcam_img = generate_gradcam(model, img_cv)
        st.image(gradcam_img, use_column_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")

    # LIME
    st.markdown("---")
    st.subheader("🧪 LIME")

    try:
        lime_img = generate_lime(model, img_cv)
        st.image(lime_img, use_column_width=True)
    except Exception as e:
        st.warning(f"LIME failed: {e}")

# -------------------------------
# HISTORY DISPLAY
# -------------------------------
st.markdown("---")
st.subheader(f"📊 {t['history']}")

for item in st.session_state.history:
    st.write(item)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
