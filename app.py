import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import os  # Ditambahkan untuk cek file

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="YOLOv8s vs GWO | Pneumonia Detection",
    page_icon="🫁",
    layout="wide" # Mengubah layout menjadi wide agar komparasi lebih jelas
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
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #00d2d3 !important;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4e4e4e;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """
    Memuat dua model sekaligus untuk komparasi dengan error handling terpisah.
    """
    path_base = 'yolov8s_base.pt'
    path_gwo = 'best.pt'

    # 1. Cek keberadaan file fisik
    if not os.path.exists(path_base):
        return None, None, f"File '{path_base}' tidak ditemukan di direktori."
    if not os.path.exists(path_gwo):
        return None, None, f"File '{path_gwo}' tidak ditemukan di direktori."

    model_base = None
    model_gwo = None

    # 2. Coba load Model Base
    try:
        model_base = YOLO(path_base)
    except Exception as e:
        return None, None, f"File '{path_base}' rusak/corrupt (Ran out of input). Ganti file ini. Detail: {e}"

    # 3. Coba load Model GWO
    try:
        model_gwo = YOLO(path_gwo)
    except Exception as e:
        return None, None, f"File '{path_gwo}' rusak/corrupt (Ran out of input). Ganti file ini. Detail: {e}"
        
    return model_base, model_gwo, None

# Memanggil fungsi load
model_base, model_gwo, error_msg = load_models()

# Cek error
if error_msg:
    st.error(f"❌ Terjadi kesalahan fatal:")
    st.error(error_msg)
    st.warning("Tips: Error 'Ran out of input' biasanya berarti file .pt tidak terunduh sempurna. Coba copy ulang file model dari sumber aslinya.")
    st.stop()

# --- HEADER ---
st.title("🫁 Komparasi Deteksi Pneumonia")
st.subheader("YOLOv8s (Standard) vs YOLOv8s + Grey Wolf Optimizer (GWO)")
st.markdown("Aplikasi ini mendemonstrasikan peningkatan performa deteksi objek setelah dilakukan optimasi hiperparameter menggunakan algoritma GWO.")

# --- SIDEBAR ---
st.sidebar.title("🎛️ Kontrol Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, help="Batas minimum keyakinan AI untuk menampilkan kotak deteksi.")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Keterangan Model:**
    - **YOLOv8s Base:** Model standar tanpa optimasi heuristic.
    - **YOLOv8s + GWO:** Model yang telah di-tuning menggunakan Grey Wolf Optimizer untuk akurasi lebih baik pada dataset X-Ray.
    """
)
st.sidebar.caption("© 2026 AzmiDev Research")

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Upload Citra X-Ray Dada", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Tampilkan Gambar Asli di Tengah (Kecil saja untuk preview)
    with st.expander("Lihat Gambar Asli", expanded=True):
        st.image(image, caption="Citra X-Ray Original", width=300)

    # Tombol Eksekusi
    if st.button('🚀 Mulai Komparasi AI'):
        
        with st.spinner('Sedang menjalankan inferensi pada kedua model...'):
            # 1. Prediksi Model Base
            res_base = model_base.predict(image, conf=conf_threshold)
            img_base = res_base[0].plot()
            img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
            boxes_base = res_base[0].boxes

            # 2. Prediksi Model GWO
            res_gwo = model_gwo.predict(image, conf=conf_threshold)
            img_gwo = res_gwo[0].plot()
            img_gwo_rgb = cv2.cvtColor(img_gwo, cv2.COLOR_BGR2RGB)
            boxes_gwo = res_gwo[0].boxes

        # --- TAMPILAN HASIL SIDE-BY-SIDE ---
        st.markdown("---")
        col1, col2 = st.columns(2)

        # === KOLOM 1: YOLOv8s BASE ===
        with col1:
            st.markdown("<h3 style='text-align: center;'>YOLOv8s (Base)</h3>", unsafe_allow_html=True)
            st.image(img_base_rgb, use_container_width=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <b>Deteksi:</b> {len(boxes_base)} objek<br>
            </div>
            """, unsafe_allow_html=True)

            if len(boxes_base) > 0:
                for box in boxes_base:
                    conf = box.conf[0]
                    cls_name = model_base.names[int(box.cls[0])]
                    st.write(f"- `{cls_name}`: **{conf:.2%}**")
            else:
                st.caption("Tidak ada deteksi.")

        # === KOLOM 2: YOLOv8s + GWO ===
        with col2:
            st.markdown("<h3 style='text-align: center; color: #00d2d3;'>YOLOv8s + GWO (Optimized)</h3>", unsafe_allow_html=True)
            st.image(img_gwo_rgb, use_container_width=True)
            
            st.markdown(f"""
            <div class='metric-card' style='border-color: #00d2d3;'>
                <b>Deteksi:</b> {len(boxes_gwo)} objek<br>
            </div>
            """, unsafe_allow_html=True)

            if len(boxes_gwo) > 0:
                for box in boxes_gwo:
                    conf = box.conf[0]
                    cls_name = model_gwo.names[int(box.cls[0])]
                    # Highlight hasil GWO
                    st.success(f"🎯 `{cls_name}`: **{conf:.2%}**")
            else:
                st.caption("Tidak ada deteksi.")

        # --- ANALISIS SINGKAT ---
        st.markdown("---")
        st.subheader("📝 Analisis Perbedaan")
        
        diff_conf = 0
        if len(boxes_base) > 0 and len(boxes_gwo) > 0:
            # Mengambil rata-rata confidence jika deteksi > 1, atau confidence tunggal jika 1
            avg_base = float(boxes_base.conf.mean())
            avg_gwo = float(boxes_gwo.conf.mean())
            diff = avg_gwo - avg_base
            
            if diff > 0:
                st.success(f"**Kesimpulan:** Model GWO memiliki rata-rata confidence **+{diff:.2%} lebih tinggi** dibandingkan model Base pada citra ini.")
            elif diff < 0:
                st.warning(f"**Kesimpulan:** Model Base memiliki rata-rata confidence **{abs(diff):.2%} lebih tinggi** (GWO mungkin mengurangi False Positives).")
            else:
                st.info("Kedua model memberikan hasil confidence yang identik.")
        elif len(boxes_gwo) > len(boxes_base):
             st.success("**Kesimpulan:** Model GWO berhasil mendeteksi objek yang terlewat oleh model Base (False Negative pada Base berkurang).")
        elif len(boxes_base) > len(boxes_gwo):
             st.info("**Kesimpulan:** Model GWO tidak mendeteksi objek yang dideteksi Base (Kemungkinan Base mendeteksi False Positive, atau GWO under-detect).")
        else:
            st.caption("Belum cukup data untuk menyimpulkan perbedaan pada citra ini.")

else:
    # Tampilan awal jika belum upload
    st.info("Silakan upload gambar X-Ray untuk melihat perbandingan kinerja model sebelum dan sesudah optimasi GWO.")
    st.markdown("---")