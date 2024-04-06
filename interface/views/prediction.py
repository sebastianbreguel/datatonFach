import streamlit as st
import leafmap.foliumap as leafmap
import rioxarray

def show_prediction():
    """
    Display the prediction of fire risk in a Streamlit page.
    """
    # tif_1 = "rasters/lansat/2024_valpo_swir16-nir-red.tif"
    predicted_tif = "rasters/predictions/valpo_prediction_2.tif"

    st.title('Predictor de incendios')

    map = leafmap.Map(latlon_control=False)
    map.split_map(left_layer="ROADMAP", right_layer=predicted_tif, left_label="Actual", right_label="Predicted")
    map.to_streamlit()
