import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Skin Type Predictor", layout="centered")
st.title("🧴 Skin Type Detection & Product Recommendation")

# =========================
# Load Dataset
# =========================
df = pd.read_excel("skincare_products_100_rows.xlsx")

# =========================
# CNN Model (Demo)
# =========================
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

skin_types = ["Oily", "Dry", "Normal", "Sensitive"]

# =========================
# Webcam Input
# =========================
img_file = st.camera_input("📸 Capture your face")

if img_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Captured Image", channels="BGR")

    # =========================
    # Preprocess Image
    # =========================
    img_resized = cv2.resize(img, (64,64))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1,64,64,3)

    # =========================
    # Predict Skin Type
    # =========================
    prediction = model.predict(img_input)
    skin_index = np.argmax(prediction)
    detected_skin = skin_types[skin_index]

    st.success(f"🧠 Detected Skin Type: **{detected_skin}**")

    # =========================
    # Product Recommendation
    # =========================
    st.subheader("🛍️ Recommended Products")
    recommended = df[df["Skin Type"].str.contains(detected_skin, case=False)]

    st.dataframe(recommended[[
        "Product Name",
        "Brand",
        "Price (₹)",
        "Reviews ⭐",
        "Low-Cost Website"
    ]].head(5))

    # =========================
    # Graph
    # =========================
    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(skin_types, prediction[0])
    ax.set_xlabel("Skin Type")
    ax.set_ylabel("Confidence")
    ax.set_title("Skin Type Prediction Confidence")
    st.pyplot(fig)
