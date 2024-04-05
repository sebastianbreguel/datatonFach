import streamlit as st
import leafmap.foliumap as leafmap
import rioxarray

def show_prediction():
    """
    Display the prediction of fire risk in a Streamlit page.
    """
    tif_1 = "../rasters/lansat/2024_valpo_swir16-nir-red.tif"
    predicted_tif = "../rasters/predictions/valpo_prediction.tif"

    st.title('Predictor de incendios')

    map = leafmap.Map(latlon_control=False)
    map.split_map(left_layer=tif_1, right_layer=predicted_tif, left_label="Actual", right_label="Predicted", left_args={"colormap": "viridis"}, right_args={"colormap": "viridis"})
    map.to_streamlit()
