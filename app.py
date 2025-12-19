import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import io
from PIL import Image

st.set_page_config(page_title="Skin Product Recommendation App", layout="wide")

st.title("🌟 AI Based Skin Product Recommendation")
st.write("Upload / Capture your face and get the best recommendation")

# ---------------- USER INPUT ----------------
name = st.text_input("Enter Your Name")

st.write("📷 Capture your Image")
img_data = st.camera_input("Take a photo")

# ------------- LOAD MODEL FROM GITHUB -------------
url = "https://raw.githubusercontent.com/USERNAME/REPO_NAME/main/cnn_model.h5"

@st.cache_resource
def load_model():
    try:
        response = requests.get(url)
        model = tf.keras.models.load_model(io.BytesIO(response.content))
        return model
    except Exception as e:
        st.error("❌ Failed to Load Model from GitHub")
        st.write(e)
        return None

model = load_model()

# ---------- PRODUCT & REVIEWS (Example) ----------
products = {
    "Oily Skin": {
        "product": "Mamaearth Oil Control Face Wash",
        "review": "4.5⭐ — Best for oil control and acne prevention.",
        "reason": "Your skin looks oily with visible shine. Oil-control products help."
    },
    "Dry Skin": {
        "product": "Cetaphil Gentle Cleanser",
        "review": "4.6⭐ — Excellent hydration and smooth skin result.",
        "reason": "Your skin appears dry & flaky. Hydrating cleanser helps moisture."
    },
    "Normal Skin": {
        "product": "Simple Refreshing Face Wash",
        "review": "4.4⭐ — Suitable for daily freshness and glow.",
        "reason": "Your skin looks balanced without excess oil or dryness."
    },
    "Acne Skin": {
        "product": "Himalaya Neem Face Wash",
        "review": "4.3⭐ — Trusted acne & pimple solution.",
        "reason": "Spots / acne detected. Neem face wash reduces bacteria."
    }
}

# ---------- PREDICTION ----------
if img_data is not None and model is not None:
    image = Image.open(img_data)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    classes = ["Oily Skin", "Dry Skin", "Normal Skin", "Acne Skin"]
    skin_type = classes[result]

    st.success(f"👤 Hello {name}, Your detected Skin Type is: **{skin_type}**")

    st.subheader("🎯 Recommended Product")
    st.write("🛍️ Product:", products[skin_type]["product"])
    st.write("⭐ Reviews:", products[skin_type]["review"])
    st.write("ℹ️ Why Suggested:", products[skin_type]["reason"])
