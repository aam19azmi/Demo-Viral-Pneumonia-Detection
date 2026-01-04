import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

st.title("🫁 Pneumonia Detection AI")
st.write("Implementasi Model YOLOv8s + GWO dari Riset Jurnal JUTIF 2025")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt') # Pastikan file best.pt ada di folder yang sama

model = load_model()

# Upload Gambar
uploaded_file = st.file_uploader("Upload X-Ray Dada (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if st.button('Mulai Deteksi'):
        # Jalankan Prediksi
        results = model.predict(image, conf=0.5)
        
        # Plot Hasil
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Hasil Deteksi Bounding Box', use_column_width=True)
        
        # Tampilkan Informasi Deteksi
        for box in results[0].boxes:
            conf = box.conf[0]
            cls = model.names[int(box.cls[0])]
            st.success(f"Terdeteksi: **{cls}** dengan Akurasi: **{conf:.2f}**")
