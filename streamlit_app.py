import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as tf_stream # İsmi çakışmasın diye st diyelim
import streamlit as st
import gdown
from PIL import Image

# Sayfa Ayarları (Tarayıcı sekmesinde görünecek kısım)
st.set_page_config(page_title="Skin Cancer AI", page_icon="🧠", layout="centered")

# 1. MODEL İNDİRME VE YÜKLEME
MODEL_PATH = "skin_cancer_model.keras"
FILE_ID = "1lX5X71ncFTlkIohrHdHcrxRu5BCcCRVF"

@st.cache_resource  # Modelin her sayfa yenilendiğinde tekrar tekrar yüklenmesini engeller
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model ilk defa indiriliyor, lütfen bekleyin..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yükleme hatası: {e}")

labels = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
    'Dermatofibroma', 'Melanoma (Tehlikeli)', 'Nevus (Ben)', 'Vascular Lesion'
]

# 2. ARAYÜZ TASARIMI
st.title("🧠 Skin Cancer Detection AI")

st.markdown("### 📤 Cilt görüntüsü yükleyerek yapay zeka ile analiz başlatabilirsiniz")
uploaded_file = st.file_uploader("Bir görsel seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", width=250)
    
    # "Tahmin Et" Butonu
    if st.button("🔍 Tahmin Et"):
        with st.spinner("Yapay zeka resmi analiz ediyor..."):
            # Görsel Ön İşleme
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # OpenCV için RGB'den BGR'ye dönüşüm
            img_resized = cv2.resize(img_bgr, (224, 224))
            img_normalized = img_resized / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)
            
            # Tahmin
            pred = model.predict(img_expanded)
            class_idx = np.argmax(pred)
            prediction = labels[class_idx]
            
            # Sonucu Yazdır
            st.success(f"### 🔍 Tahmin Sonucu: **{prediction}**")
            st.warning("⚠️ Bu sonuç yapay zeka modelinin tahminidir. Kesin tıbbi teşhis için uzman doktora başvurulmalıdır.")

# PROJE BİLGİLERİ (Her zaman altta görünecek sabit kısımlar)
st.write("---")
st.markdown("""
## 📘 Proje Hakkında
* **Problem:** Dermatoskopik görüntüler üzerinden cilt kanseri türlerinin otomatik sınıflandırılması.
* **Amaç:** Yapay zeka ile hızlı ve doğru hastalık tahmini yapmak.
* **Önemi:** Erken teşhis hayat kurtarır. Bu sistem ön değerlendirme sağlar.
* **Veri Seti:** HAM10000 veri seti (10015 görüntü, 7 sınıf)

## 📂 Veri Seti ve Ön İşleme
* 224x224 boyutlandırma
* Normalization (0-1)
* Train/Test ayrımı

## 🧠 Model Bilgisi
* CNN modeli kullanıldı
* Conv2D, MaxPooling, Dense katmanları
* Epoch: 10 | Batch: 32 | Optimizer: Adam
""")

# Grafikler (Eğer klasöründe bu görseller varsa gösterilir, yoksa hata vermez)
st.markdown("## 📊 Grafikler")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("static/confusion_matrix.png"): st.image("static/confusion_matrix.png", caption="Confusion Matrix")
    if os.path.exists("static/accuracy.png"): st.image("static/accuracy.png", caption="Accuracy")
with col2:
    if os.path.exists("static/roc_curve.png"): st.image("static/roc_curve.png", caption="ROC Curve")
    if os.path.exists("static/loss.png"): st.image("static/loss.png", caption="Loss")

st.markdown("""
## 📈 Performans Metrikleri
* **Accuracy:** %87
* **Precision:** %85
* **Recall:** %83
* **F1 Score:** %84
* **AUC:** %90

## 📚 Kaynakça
* [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* [TensorFlow](https://www.tensorflow.org/)
""")