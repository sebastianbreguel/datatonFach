import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
import numpy as np


# Calcular la cantidad de hectáreas quemadas
def calcular_hectareas_quemadas(src_img):
    # Leer los datos del raster como una matriz NumPy
    raster_data = src_img.read(1)
    threshold = 0  # Umbral para considerar un píxel como quemado

    # Calcular la cantidad de píxeles quemados
    pixeles_quemados = np.sum(raster_data > threshold)
    tamanio_pixel = src_img.transform.a * src_img.transform.e
    hectareas_quemadas = pixeles_quemados * tamanio_pixel / 10000

    return hectareas_quemadas


tif = "C:/Users/lucas/OneDrive/Documentos/GitHub/datatonFach/interface/rasters/lansat/dnbr.tif"
src_img = rasterio.open(tif)

st.set_page_config(page_title="Cuantificación de incendios", layout="wide")
st.title('Cuantificación de incendios incendio')

row1_col1, row1_col2 = st.columns([5, 2])

with row1_col1:
    map = leafmap.Map(latlon_control=False)
    map.add_raster(tif, colormap="viridis", layer_name="Landsat")
    map.to_streamlit()

with row1_col2:
    st.write("## Métricas")
    hectareas_quemadas = calcular_hectareas_quemadas(src_img)
    row1_col2_col1, row1_col2_col2 = st.columns([1, 3])
    with row1_col2_col1:
        st.image("img/terreno.png")
    with row1_col2_col2:
        st.metric(label="Hectáreas quemadas", value=hectareas_quemadas, delta=None)

