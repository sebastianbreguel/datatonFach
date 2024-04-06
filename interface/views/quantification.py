import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
import numpy as np
import os


def procesar_imagenes(pre_fire_src, post_fire_src):
    """
    Calculate the number of burned hectares from a given raster image.

    Parameters:
    src_img (rasterio.io.DatasetReader): The raster image dataset reader.

    Returns:
    float: The number of burned hectares.
    """
    nbr_pre_fire = (pre_fire_src.read(3) - pre_fire_src.read(2)) / (
        pre_fire_src.read(3) + pre_fire_src.read(2)
    )
    nbr_post_fire = (post_fire_src.read(3) - post_fire_src.read(2)) / (
        post_fire_src.read(3) + post_fire_src.read(2)
    )
    dnbr = nbr_pre_fire - nbr_post_fire

    dnbr_ranges = {
        "enhaced_regrowth_high": (-0.500, -0.251),
        "enhaced_regrowth_low": (-0.250, -0.101),
        "unburned": (-0.100, 0.99),
        "low_severity": (0.100, 0.269),
        "moderate_low_severity": (0.270, 0.439),
        "moderate_high_severity": (0.440, 0.659),
        "high_severity": (0.660, 1.300),
    }
    dnbr_gray_values = {
        "enhaced_regrowth_high": 0,
        "enhaced_regrowth_low": 1,
        "unburned": 2,
        "low_severity": 3,
        "moderate_low_severity": 4,
        "moderate_high_severity": 5,
        "high_severity": 6,
    }
    dnbr_counts = {
        "enhaced_regrowth_high": 0,
        "enhaced_regrowth_low": 0,
        "unburned": 0,
        "low_severity": 0,
        "moderate_low_severity": 0,
        "moderate_high_severity": 0,
        "high_severity": 0,
    }
    for key, value in dnbr_ranges.items():
        dnbr_counts[key] = ((dnbr >= value[0]) & (dnbr <= value[1])).sum()
    dnbr_counts = {
        key: ((int(value) * 30 * 30) / 10000) for key, value in dnbr_counts.items()
    }

    dnbr_mask = np.zeros(dnbr.shape, dtype=np.uint8)
    for key, value in dnbr_ranges.items():
        dnbr_mask[(dnbr >= value[0]) & (dnbr <= value[1])] = dnbr_gray_values[key]

    dnbr_mask[np.logical_not(dnbr_mask > 0)] = 0

    profile = pre_fire_src.profile
    profile.update(dtype=rasterio.float32, count=1)

    path_dnbr_2023_2024_discretised = "rasters/lansat/dnbr_2023_2024_discretised.tif"
    if os.path.exists(path_dnbr_2023_2024_discretised):
        os.remove(path_dnbr_2023_2024_discretised)
    with rasterio.open(
        "rasters/lansat/dnbr_2023_2024_discretised.tif", "w", **profile
    ) as dst:
        dst.write(dnbr_mask, 1)

    return dnbr_counts


def show_quantification():
    """
    Display the quantification of fires in a Streamlit page.
    """
    tif = "rasters/lansat/dnbr_2023_2024_discretised.tif"
    tif_pre_fire = "rasters/lansat/2023_valpo_swir16-nir-red.tif"
    tif_post_fire = "rasters/lansat/2024_valpo_swir16-nir-red.tif"

    pre_fire = rasterio.open(tif_pre_fire)
    post_fire = rasterio.open(tif_post_fire)

    dnbr_counts_pixels = procesar_imagenes(pre_fire, post_fire)

    st.title("Cuantificador de incendios")

    row1_col1, row1_col2 = st.columns([5, 2])

    with row1_col1:
        map = leafmap.Map(latlon_control=False)
        # Mapas de color disponibles en: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        map.add_raster(
            tif, layer_name="Landsat", colormap="Dark2", opacity=0.8, nodata=0
        )
        map.to_streamlit()

    with row1_col2:
        st.write("## Métricas")
        hectareas_quemadas = (
            dnbr_counts_pixels["high_severity"]
            + dnbr_counts_pixels["moderate_high_severity"]
            + dnbr_counts_pixels["moderate_low_severity"]
            + dnbr_counts_pixels["low_severity"]
        )

        row1_col2_col1, row1_col2_col2 = st.columns([1, 3])
        with row1_col2_col1:
            st.image("img/terreno.png")
        with row1_col2_col2:
            st.metric(
                label="Hectáreas quemadas", value=f"{hectareas_quemadas}", delta=None
            )

        # Show the different categories of burned hectares
        st.write("##### Intencidad de incendio (en hectáreas):")
        st.write(f"🟨 Baja intensidad: {dnbr_counts_pixels['low_severity']}")
        st.write(
            f"🟧 Media intensidad {(dnbr_counts_pixels['moderate_high_severity'] + dnbr_counts_pixels['moderate_low_severity'])}"
        )
        st.write(f"🟥 Alta intensidad: {dnbr_counts_pixels['high_severity']}")
