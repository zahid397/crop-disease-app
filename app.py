import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crop Disease Doctor",
    page_icon="🍃",
    layout="centered"
)

# ---------------- UI Style ----------------
st.markdown("""
<style>
.title {
    text-align: center;
    color: #2E7D32;
    font-size: 3em;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #4CAF50;
    font-size: 1.2em;
    margin-bottom: 20px;
}
.result-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="title">🍃 Crop Disease Doctor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI Powered Crop Disease Detection for Sustainable Agriculture</div>',
    unsafe_allow_html=True
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5")

@st.cache_resource
def load_classes():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model()
    class_names = load_classes()
except:
    st.error("❌ Model or class_names file missing!")
    st.stop()

# ---------------- Image Upload ----------------
st.divider()
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Detect Disease", type="primary"):
        with st.spinner("Analyzing leaf pattern... 🌿"):
            # -------- Preprocessing --------
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # -------- Prediction --------
            predictions = model.predict(img_array)
            index = np.argmax(predictions)
            confidence = float(np.max(predictions))
            label = class_names[index]

            # -------- Output --------
            st.divider()
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            if "Healthy" in label:
                st.success(f"✅ Status: **{label}**")
            else:
                st.error(f"🦠 Disease Detected: **{label}**")

            st.progress(int(confidence * 100))
            st.write(f"**Confidence:** {confidence:.2%}")

            st.markdown('</div>', unsafe_allow_html=True)

            st.caption(
                "⚠️ This AI result is for educational purposes. "
                "Please consult an agriculture expert for final decisions."
            )

# ---------------- Footer ----------------
st.divider()
st.markdown(
    "<center>🌱 Developed by Zahid Hasan | AI for Agriculture 🇧🇩</center>",
    unsafe_allow_html=True
)
