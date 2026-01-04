import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Pneumonia Detection AI | AzmiDev",
    page_icon="🫁",
    layout="centered"
)

# --- STYLE CSS CUSTOM ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00d2d3;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Menggunakan YOLOv8s yang sudah di-tuning dengan GWO sesuai jurnal
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model 'best.pt'. Pastikan file tersedia. Error: {e}")

# --- HEADER ---
st.title("🫁 Pneumonia Detection AI")
st.subheader("Implementasi YOLOv8s + GWO")
st.info("Berdasarkan Riset Jurnal: *Interpretable Hybrid YOLOv8s-GWO Framework for Bounding-Box Viral Pneumonia Detection*")

# --- SIDEBAR INFO ---
st.sidebar.title("Informasi Riset")
st.sidebar.write("**Peneliti:** Azmi Jalaluddin Amron")
st.sidebar.write("**Dataset:** Kaggle Chest X-ray Images")
st.sidebar.write("**Model:** YOLOv8s (Small) + Grey Wolf Optimizer")
st.sidebar.markdown("---")
st.sidebar.write("Demo ini bertujuan untuk mendeteksi keberadaan Pneumonia pada citra X-ray dada secara real-time.")

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar X-Ray Dada...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Gambar Asli")
        st.image(image, use_container_width=True)

    # Tombol Deteksi
    if st.button('Mulai Analisis AI'):
        with st.spinner('AI sedang menganalisis pola infeksi...'):
            # Jalankan Inferensi
            # conf=0.25 (default), sesuaikan dengan hasil training terbaikmu
            results = model.predict(image, conf=0.40)
            
            # Ambil gambar hasil plotting
            res_plotted = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.write("### Hasil Deteksi")
                st.image(res_plotted_rgb, use_container_width=True)

        # --- TAMPILKAN HASIL DATA ---
        st.markdown("---")
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.warning(f"Terdeteksi {len(boxes)} area indikasi Pneumonia.")
            for box in boxes:
                confidence = box.conf[0]
                label = model.names[int(box.cls[0])]
                st.write(f"🔍 **Label:** `{label}` | **Tingkat Keyakinan (Confidence):** `{confidence:.2%}`")
        else:
            st.success("Tidak ditemukan indikasi Pneumonia pada tingkat kepercayaan > 40%.")

        # --- FITUR DOWNLOAD ---
        # Mengubah hasil ke format yang bisa didownload
        result_img = Image.fromarray(res_plotted_rgb)
        st.markdown("### Simpan Hasil")
        
        # Simpan ke buffer memory
        import io
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Hasil Analisis (.png)",
            data=byte_im,
            file_name="hasil_deteksi_pneumonia.png",
            mime="image/png"
        )

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Azmi Jalaluddin Amron | Source Code Journal Tech. Hanya untuk tujuan demonstrasi riset.")
