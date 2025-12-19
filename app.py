# app.py - Streamlit version
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import base64
import io
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="Skincare Product Predictor", page_icon="🧴", layout="wide")

# Load model and data
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

@st.cache_data
def load_products():
    df = pd.read_excel('skin_products-1.xlsx')
    return df

model = load_model()
df = load_products()

# Skin type mapping
skintypes = ['Oily', 'Dry', 'Normal', 'Sensitive']

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #FF69B4; text-align: center; margin-bottom: 2rem;}
    .skin-result {font-size: 2rem; color: #4CAF50; text-align: center; padding: 1rem; border-radius: 10px;}
    .product-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧴 Skincare Product Predictor</h1>', unsafe_allow_html=True)

# Sidebar for instructions
with st.sidebar:
    st.header("📋 How to use")
    st.write("""
    1. **Allow camera access**
    2. **Take a photo** of your skin (clean, no makeup)
    3. **Get personalized recommendations**
    4. **Click product links** to buy
    """)
    st.info("Best results: Take photo in natural light, close-up of cheek/forehead area")

# Image capture component
def get_base64_image():
    return components.html("""
    <div style="text-align: center;">
        <video id="video" width="400" height="300" autoplay muted></video>
        <br><br>
        <button id="capture" style="padding: 10px 20px; font-size: 16px; background: #FF69B4; color: white; border: none; border-radius: 5px; cursor: pointer;">📸 Capture Photo</button>
        <br><br>
        <canvas id="canvas" width="400" height="300" style="display: none;"></canvas>
        <br>
        <img id="photo" width="400" height="300" style="border-radius: 10px; display: none;">
    </div>
    <script>
    const video = document.getElementById('video');
    const capture = document.getElementById('capture');
    const canvas = document.getElementById('canvas');
    const photo = document.getElementById('photo');
    
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
        .then(stream => {
            video.srcObject = stream;
        });
    
    capture.addEventListener('click', () => {
        canvas.getContext('2d').drawImage(video, 0, 0, 400, 300);
        const dataUrl = canvas.toDataURL('image/jpeg');
        photo.src = dataUrl;
        photo.style.display = 'block';
        video.style.display = 'none';
        capture.style.display = 'none';
        
        // Send to Streamlit
        parent.document.querySelector('iframe').contentWindow.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: dataUrl
        }, '*');
    });
    </script>
    """, height=500)

# Main app
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload or Capture")
    
    # Option 1: Webcam
    webcam_data = get_base64_image()
    
    # Option 2: File upload
    uploaded_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

# Process image and predict
if st.button("🔍 Analyze Skin Type") or st.session_state.prediction_made:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        # Use webcam image (simplified for demo - in production use the component callback)
        st.warning("Please upload an image for analysis")
        st.stop()
    
    # Preprocess
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    with st.spinner("Analyzing your skin..."):
        prediction = model.predict(img, verbose=0)
        skin_index = np.argmax(prediction[0])
        confidence = prediction[0][skin_index]
        detected_skin = skintypes[skin_index]
    
    st.session_state.prediction_made = True
    
    # Display result
    st.markdown(f"""
    <div class="skin-result">
        Your Skin Type: **{detected_skin}** ({confidence:.1%} confidence)
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("🥗 Personalized Product Recommendations")
    
    recommended = df[df['Skin Type'].str.contains(detected_skin, case=False, na=False)].head(10)
    
    if not recommended.empty:
        for idx, row in recommended.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['Product Name']}**")
                    st.caption(f"{row['Brand']} | ⭐ {row['Reviews']}")
                with col2:
                    st.metric("₹", row['Price'])
                with col3:
                    if st.button(f"🛒 Buy", key=f"buy_{idx}"):
                        st.info(f"Available at {row['Low-Cost Website']}")
    else:
        st.warning("No products found for this skin type.")

# Footer
st.markdown("---")
st.markdown("💡 *Powered by AI Skin Analysis | Always patch test new products*")


