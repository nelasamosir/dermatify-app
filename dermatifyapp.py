import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ======== Konfigurasi halaman Streamlit ========
st.set_page_config(layout="wide")

# ======== Load model CNN ========
@st.cache_resource
def load_cnn_model():
    return load_model("Model_Skenario 3_Adam_R80.h5")

model = load_cnn_model()

# ======== Label dan Deskripsi Penyakit ========
labels = ['Melanoma', 'Eczema', 'Benign Keratosis', 'Melanocytic Nevi', 'Basal Cell Carcinoma']
deskripsi_penyakit = {
    "Melanoma": "Melanoma adalah tumor ganas dari sel melanosit dan terutama terjadi di kulit. Melanoma juga dapat timbul di mata (uvea, konjungtiva dan tubuh ciliary), meninges, dan mukosa permukaan tubuh yang mengandung melanin. Melanoma menyumbang 90 persen dari kematian terkait tumor kulit.",
    "Eczema": "Eczema atau Eksim merupakan kondisi kulit dimana disebabkan oleh peradangan kulit. Eksim yang merupakan sejenis alergi ini meningkat selama lebih dari 2 dekade pada negara berindustri, seperti Indonesia . Eksim di sini merupakan inflamasi kronis yang ditandai dengan ruam merah gatal yang menyokong lipatan kulit seperti lipatan siku atau di belakang lutut.",
    "Benign Keratosis": "Benign Keratosis atau Keratosis seboroik (SK) juga dikenal sebagai kutil seboroik dan papiloma sel basal. Ini adalah pertumbuhan jinak yang disebabkan oleh penumpukan sel kulit. SK sangat umum, tidak berbahaya, seringkali berwarna coklat atau hitam, dan muncul di kulit",
    "Melanocytic Nevi": "Melanocytic Nevi atau Nevus melanositik (MN), yang biasanya disebut sebagai tahi lalat, adalah neoplasma jinak yang berasal dari sel nevus asal puncak saraf. Selain menetapkan asal embrionik mereka dari puncak saraf, histogenesis tumor-tumor ini tetap menjadi subjek spekulasi dan sudut pandang yang beragam.",
    "Basal Cell Carcinoma": "Basal Cell Carcinoma atau KSB adalah tumor ganas yang bersifat invasif secara lokal, agresif, dan destruktif. Etiopatogenesis KSB adalah predisposisi genetik, lingkungan, dan paparan sinar matahari, khususnya ultraviolet B (UVB) yang merangsang terjadinya mutasi suppressor genes. malignansi ini biasanya timbul di daerah yang terpapar sinar matahari."
}


# ======== Fungsi Deteksi Kulit ========
def contains_skin(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Rentang warna kulit (bisa disesuaikan)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > 0.02  # minimal 2% area ada kulit

# ======== Fungsi Prediksi ========
def predict_image(image):
    if not contains_skin(image):
        return "Tidak terdefinisi", 0.0

    image = image.convert("RGB").resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0]
    idx = np.argmax(pred)
    return labels[idx], round(pred[idx] * 100, 2)

# ======== Styling ========
def local_css():
    st.markdown("""
    <style>
        .title { font-size: 40px; font-weight: bold; color: #2DC8C8; text-align: center; }
        .subtitle { font-size: 24px; font-weight: 600; margin-top: 10px; }
        .desc-box { background-color: white; padding: 15px; border-radius: 10px; font-size: 16px; text-align: justify; }
        .blue-box { background-color: #d9f9f9; padding: 10px 20px; border-radius: 10px; margin-top: 20px; text-align: center; }
        .centered { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
        .divider { width: 1px; background-color: #CCCCCC; min-height: 650px; margin: auto; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ======== Layout Utama ========
col1, spacer1, col2, spacer2, col3 = st.columns([1.1, 0.05, 1.5, 0.05, 1.5])

# Kolom 1
with col1:
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.markdown('<div class="title">DERMATIFY</div>', unsafe_allow_html=True)
    st.image("assets/doc1.png", width=280)
    st.markdown('<div class="blue-box">Empowering you to identify<br>and understand your skin</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Divider
with spacer1:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Kolom 2
with col2:
    st.markdown('<div class="subtitle">DERMATIFY LAB</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Gambar Kulit", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        hasil, akurasi = predict_image(image)
    else:
        hasil, akurasi = None, None

# Divider
with spacer2:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Kolom 3
with col3:
    st.markdown('<div class="subtitle">DESKRIPSI PENYAKIT KULIT</div>', unsafe_allow_html=True)

    if hasil:
        st.markdown(f"**{hasil}**<br>Akurasi : {akurasi} %", unsafe_allow_html=True)
        st.markdown(f'<div class="desc-box">{deskripsi_penyakit.get(hasil, "Deskripsi tidak tersedia")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='desc-box'>Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    st.markdown('<div class="subtitle" style="margin-top: 20px;">About Dermatify</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="desc-box">
            Aplikasi cerdas berbasis AI yang membantu mengenali jenis penyakit kulit hanya melalui gambar. 
            Dengan teknologi CNN (Convolutional Neural Network) yang telah dilatih khusus untuk mampu memberikan 
            prediksi cepat dan akurat untuk lima jenis kondisi masalah kulit.
        </div>
    """, unsafe_allow_html=True)
