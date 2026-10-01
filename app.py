import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuración inicial de la página
st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="📷",
    layout="centered"
)

# Estilos CSS personalizados (Fondo morado claro y diseño moderno)
st.markdown("""
    <style>
    /* Fondo principal morado claro */
    .stApp {
        background-color: #f3e9f8;
    }
    
    /* Contenedor lateral */
    [data-testid="stSidebar"] {
        background-color: #e3d2f2;
    }

    /* Títulos e instrucciones */
    h1 {
        color: #4a154b;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    h3 {
        color: #5c2c69;
    }

    /* Caja de predicción estilo tarjeta */
    .prediction-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        border-left: 6px solid #8a2be2;
    }
    .prediction-title {
        font-size: 22px;
        color: #4a154b;
        font-weight: bold;
    }
    .prediction-prob {
        font-size: 18px;
        color: #6a1b9a;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("📷 Reconocimiento de Imágenes")

# Barra lateral informativa
with st.sidebar:
    st.title("💡 Información")
    st.write("Esta aplicación utiliza un modelo entrenado en **Teachable Machine** para clasificar imágenes capturadas con tu cámara en tiempo real.")
    st.info("Asegúrate de permitir el acceso a tu cámara para continuar.")

# Cargar imagen decorativa si existe
try:
    image = Image.open('OIG5.jpg')
    st.image(image, use_column_width=True)
except Exception:
    pass

# Cargar el modelo guardado de Teachable Machine
@st.cache_resource
def cargar_modelo():
    try:
        return load_model('keras_model.h5', compile=False)
    except Exception as e:
        st.error("Error al cargar el modelo 'keras_model.h5'. Asegúrate de que está subido a la raíz de tu repositorio.")
        return None

model = cargar_modelo()

# Entrada de la cámara
img_file_buffer = st.camera_input("Toma una foto para clasificar")

if img_file_buffer is not None and model is not None:
    # 1. Cargar la imagen desde la cámara
    img = Image.open(img_file_buffer)

    # 2. Redimensionar a (224, 224) para Teachable Machine
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # 3. Convertir a array de NumPy
    img_array = np.asarray(img)

    # 4. Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1.0

    # 5. Crear el batch de 1 elemento
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 6. Realizar la predicción
    prediction = model.predict(data)
    
    # 7. Definir las clases de tu modelo
    # Modifica o agrega más nombres de etiquetas según las clases de tu Teachable Machine
    labels = ["Izquierda", "Arriba", "Derecha"]

    # Obtener el índice con mayor probabilidad
    max_index = np.argmax(prediction[0])
    probabilidad = prediction[0][max_index] * 100
    clase_detectada = labels[max_index] if max_index < len(labels) else f"Clase {max_index}"

    # Mostrar resultado en tarjeta visual
    st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">Resultado: {clase_detectada}</div>
            <div class="prediction-prob">Confianza / Probabilidad: {probabilidad:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

