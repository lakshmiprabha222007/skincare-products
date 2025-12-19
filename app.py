import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from PIL import Image
import io
import tempfile

# Page config
st.set_page_config(page_title="🧴 Skincare Predictor", page_icon="🧴", layout="wide")

# 🔥 UPDATE THESE WITH YOUR GITHUB REPO LINKS
YOUR_USERNAME = "YOUR_USERNAME"  # ← CHANGE THIS!
MODEL_URL = f"https://github.com/lakshmiprabha222007/skincare-products/blob/main/cnn_model.h5"
DATA_URL = f"https://github.com/{YOUR_USERNAME}/skincare-app/raw/main/skin_products-1.xlsx"

@st.cache_resource
def load_model_from_github():
    """Download H5 model from GitHub"""
    try:
        with st.spinner("🔄 Downloading AI model..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.h5').name
            with open(model_path, 'wb') as f:
                f.write(response.content)
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model download failed: {str(e)}")
        st.stop()

@st.cache_data
def load_products_from_github():
    """Download products data"""
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        return df
    except Exception as e:
        st.error(f"❌ Data download failed: {str(e)}")
        st.stop()

# Load everything
st.title("🧴 AI Skincare Product Recommender")
st.markdown("---")

model = load_model_from_github()
df = load_products_from_github()

skintypes = ['Oily', 'Dry', 'Normal', 'Sensitive']

# Sidebar
with st.sidebar:
    st.header("📊 Your GitHub Links")
    st.success(f"**Model**: [model.h5]({MODEL_URL})")
    st.success(f"**Products**: [Excel]({DATA_URL})")
    st.markdown("---")
    st.info("👉 Upload clean skin photo (no makeup)")

# Main app
uploaded_file = st.file_uploader("📸 Upload Skin Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your skin photo", use_column_width=True)
    
    if st.button("🔍 **Analyze My Skin Type**", type="primary"):
        with st.spinner("🤖 AI analyzing..."):
            # Preprocess image
            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (64, 64))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            prediction = model.predict(img, verbose=0)
            skin_index = np.argmax(prediction[0])
            confidence = prediction[0][skin_index]
            detected_skin = skintypes[skin_index]
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Detected Skin Type", detected_skin)
        with col2:
            st.metric("📈 Confidence", f"{confidence:.1%}")
        
        # Recommendations
        st.subheader(f"🥗 **Top {detected_skin} Skin Products**")
        recommended = df[df['Skin Type'].str.contains(detected_skin, case=False, na=False)].head(10)
        
        if not recommended.empty:
            for idx, row in recommended.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**{row['Product Name']}**")
                        st.caption(f"{row['Brand']} | ⭐ {row['Reviews']}")
                    with col2:
                        st.metric("₹", row['Price'])
                    with col3:
                        st.markdown(f"[🛒 Buy](https://www.amazon.in/s?k={row['Product Name']}+{row['Brand']})")
        else:
            st.warning("No products found for this skin type.")

st.markdown("---")
st.markdown("*💡 Always patch test new products | Made with ❤️ by AI*")

