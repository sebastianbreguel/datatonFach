import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
import numpy as np

def calcular_hectareas_quemadas(src_img: rasterio.io.DatasetReader) -> float:
    """
    Calculate the number of burned hectares from a given raster image.
    
    Parameters:
    src_img (rasterio.io.DatasetReader): The raster image dataset reader.
    
    Returns:
    float: The number of burned hectares.
    """
    tif_pre_fire = "../rasters/lansat/2023_valpo_swir16-nir-red.tif"
    tif_post_fire = "../rasters/lansat/2024_valpo_swir16-nir-red.tif"

    pre_fire = rasterio.open(tif_pre_fire)
    post_fire = rasterio.open(tif_post_fire)
    
    

def show_quantification():
    """
    Display the quantification of fires in a Streamlit page.
    """
    tif = "rasters/lansat/dnbr_2023_2024_discretised.tif"
    src_img = rasterio.open(tif)

    st.title('Cuantificador de incendios')

    row1_col1, row1_col2 = st.columns([5, 2])

    with row1_col1:
        map = leafmap.Map(latlon_control=False)
        # Mapas de color disponibles en: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        map.add_raster(tif, layer_name="Landsat", colormap="RdBu", opacity=0.7, nodata=0)
        map.to_streamlit()

    with row1_col2:
        st.write("## Métricas")
        hectareas_quemadas = 1000
        row1_col2_col1, row1_col2_col2 = st.columns([1, 3])
        with row1_col2_col1:
            st.image("img/terreno.png")
        with row1_col2_col2:
            st.metric(label="Hectáreas quemadas", value=f"{hectareas_quemadas:.2f}", delta=None)

        # Show the different categories of burned hectares
        st.write("##### Categorias de hectáreas quemadas")
        st.write(f"🟨 Baja intensidad: {hectareas_quemadas * 0.4}")
        st.write(f"🟧 Media intensidad {hectareas_quemadas * 0.1}")
        st.write(f"🟥 Alta intensidad: {hectareas_quemadas * 0.5}")
