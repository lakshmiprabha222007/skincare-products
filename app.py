import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import io
import tempfile
from PIL import Image

st.set_page_config(page_title="Skin Product Recommendation App", layout="wide")

st.title("🌟 AI Based Skin Product Recommendation")
st.write("Upload / Capture your face and get the best recommendation")

# -------- USER INPUT --------
name = st.text_input("Enter Your Name")
img_data = st.camera_input("📷 Take a Photo")

# -------- GITHUB MODEL URL --------
url = "https://raw.githubusercontent.com/USERNAME/REPO_NAME/main/cnn_model.h5"

@st.cache_resource
def load_model():
    try:
        st.info("Downloading Model from GitHub...")

        response = requests.get(url)

        if response.status_code != 200:
            st.error("❌ Model file not found in GitHub")
            return None
       
        # Save model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            temp_path = tmp.name
       
        model = tf.keras.models.load_model(temp_path)
        st.success("✅ Model Loaded Successfully!")
        return model

    except Exception as e:
        st.error("❌ Failed to Load Model")
        st.write(e)
        return None

model = load_model()

# -------- PRODUCTS DATA --------
products = {
    "Oily Skin": {
        "product": "Mamaearth Oil Control Face Wash",
        "review": "4.5⭐ — Best for oil control & acne.",
        "reason": "Detected excess oil. Oil-control products reduce acne."
    },
    "Dry Skin": {
        "product": "Cetaphil Gentle Cleanser",
        "review": "4.6⭐ — Great hydration for dry skin.",
        "reason": "Skin looks dry. Hydration protects moisture barrier."
    },
    "Normal Skin": {
        "product": "Simple Refreshing Face Wash",
        "review": "4.4⭐ — Perfect for daily gentle cleansing.",
        "reason": "Balanced texture. Gentle cleanser is ideal."
    },
    "Acne Skin": {
        "product": "Himalaya Neem Face Wash",
        "review": "4.3⭐ — Trusted acne control.",
        "reason": "Spots detected. Neem helps reduce bacteria."
    }
}

# -------- PREDICT --------
if img_data is not None and model is not None:
    image = Image.open(img_data)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    classes = ["Oily Skin", "Dry Skin", "Normal Skin", "Acne Skin"]
    skin_type = classes[result]

    st.success(f"👤 Hello {name}, Your Skin Type: **{skin_type}**")

    st.subheader("🎯 Recommended Product")
    st.write("🛍️ Product:", products[skin_type]["product"])
    st.write("⭐ Reviews:", products[skin_type]["review"])
    st.write("ℹ️ Why Suggested:", products[skin_type]["reason"])
