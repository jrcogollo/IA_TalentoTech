#Para Ejecutar: streamlit run AzureOCRStramlitTalentoTech.py

import streamlit as st
from PIL import Image
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time

# Configuración de Azure Computer Vision
VISION_ENDPOINT = "https://SU_Instancia.cognitiveservices.azure.com/"
VISION_KEY = "Su KEY"

st.title("🖼️ OCR con Azure Computer Vision")

st.write("Sube una imagen para extraer el texto mediante Azure OCR:")

uploaded_file = st.file_uploader("Elige una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    st.write("Procesando OCR con Azure...")

    # Convertir imagen a bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Cliente de Computer Vision
    computervision_client = ComputerVisionClient(
        VISION_ENDPOINT,
        CognitiveServicesCredentials(VISION_KEY)
    )

    # Llamar a la API Read (OCR)
    read_response = computervision_client.read_in_stream(BytesIO(img_byte_arr), raw=True)

    # Obtener ID de operación
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    # Esperar resultado
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Mostrar resultado
    if read_result.status == OperationStatusCodes.succeeded:
        extracted_text = ""
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                extracted_text += line.text + "\n"

        st.subheader("Texto extraído:")
        st.text_area("Resultado OCR:", extracted_text, height=250)
    else:
        st.error("No se pudo extraer texto.")

