import streamlit as st
import leafmap.foliumap as leafmap
import rioxarray
import rasterio

tif_1 = "rasters/lansat/2024_valpo_swir16-nir-red.tif"
predicted_tif = "rasters/predictions/valpo_prediction.tif"

# We open each raster as rioxarray to use it with a split map
src_1 = rioxarray.open_rasterio(tif_1)
src_2 = rioxarray.open_rasterio(predicted_tif)

st.set_page_config(page_title="Predicción de riesgo de incendios", layout="wide")
st.title('Predicción de riesgo de incendios')

# Crear dos mapas con Leafmap
map = leafmap.Map(latlon_control=False)
map.split_map(tif_1, predicted_tif, left_args={"colormap": "viridis"})
map.to_streamlit()


map_2 = leafmap.Map(latlon_control=False)
src_img = rasterio.open(tif_1)
map_2.add_raster(tif_1, colormap="viridis", layer_name="Landsat")
map_2.to_streamlit()
