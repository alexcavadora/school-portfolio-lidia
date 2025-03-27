import streamlit as st
from streamlit_drawable_canvas import st_canvas as st_canvas_component
import requests
from PIL import Image
import io

API_ENDPOINT = 'http://126.0.0.1:300/predict'

def st_canvas(fill_color, stroke_width, stroke_color, background_color, height, width, drawing_mode, key):
    return st_canvas_component(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        key=key
    )

st.title("Reconocimiento de digitos")
canvas = st_canvas(
    fill_color='white',
    stroke_width=20,
    background_color='black',
    stroke_color='white',
    height=500,
    width=500,
    drawing_mode='freedraw',
    key='canvas',
    )

if st.button('Predict'):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype('uint8'))
        img = img.convert('L')
        img = img.resize((28,28))
        imgB = io.BytesIO()
        img.save(imgB, format='PNG')
        imgB = imgB.getvalue()

        files = {"file": ('image.png',imgB,"image/png")}
        response = requests.post(API_ENDPOINT, files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            st.write(f"prediction: {prediction}")
        else: st.error(f"Error al enviar la imagen{response.status_code}")
    else:
        st.warning("Dibuja un numero!")