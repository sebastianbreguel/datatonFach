import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
import numpy as np


def calcular_hectareas_quemadas(
    pre_fire_src: rasterio.io.DatasetReader, post_fire_src: rasterio.io.DatasetReader
) -> dict:
    """
    Calculate the burned area in hectares from pre and post-fire satellite images, categorized by severity.

    Parameters:
    pre_fire_src (rasterio.io.DatasetReader): Pre-fire raster image dataset reader.
    post_fire_src (rasterio.io.DatasetReader): Post-fire raster image dataset reader.

    Returns:
    Dict[str, float]: Dictionary with the area of burned hectares categorized by severity.
    """
    nbr_pre_fire = (pre_fire_src.read(2) - pre_fire_src.read(1)) / (
        pre_fire_src.read(2) + pre_fire_src.read(1)
    )
    nbr_post_fire = (post_fire_src.read(2) - post_fire_src.read(1)) / (
        post_fire_src.read(2) + post_fire_src.read(1)
    )
    dnbr = nbr_pre_fire - nbr_post_fire

    # Print min and max values without considering nodata values
    dnbr = np.ma.masked_equal(dnbr, 0)

    dnbr_ranges = {
        "enhanced_regrowth_high": (-0.500, -0.251),
        "enhanced_regrowth_low": (-0.250, -0.101),
        "unburned": (-0.100, 0.099),  # Corrected range
        "low_severity": (0.100, 0.2),
        "moderate_low_severity": (0.2, 0.3),
        "moderate_high_severity": (0.3, 0.4),
        "high_severity": (0.4, 1.1),
    }

    dnbr_counts = {key: 0 for key in dnbr_ranges}
    for key, (lower_bound, upper_bound) in dnbr_ranges.items():
        dnbr_counts[key] = np.sum((dnbr >= lower_bound) & (dnbr <= upper_bound))

    # Convert pixel count to hectares assuming each pixel is 30x30 meters.
    pixel_area_hectares = 30 * 30 / 10000
    dnbr_areas = {
        key: count * pixel_area_hectares for key, count in dnbr_counts.items()
    }

    return dnbr_areas


def show_quantification():
    """
    Display the quantification of fires in a Streamlit page.
    """
    tif = "rasters/lansat/dnbr_2023_2024_discretised.tif"
    tif_pre_fire = "rasters/lansat/2023_valpo_swir16-nir-red.tif"
    tif_post_fire = "rasters/lansat/2024_valpo_swir16-nir-red.tif"

    pre_fire = rasterio.open(tif_pre_fire)
    post_fire = rasterio.open(tif_post_fire)

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
        dnbr_counts_pixels = calcular_hectareas_quemadas(pre_fire, post_fire)
        hectareas_quemadas = (
            dnbr_counts_pixels["high_severity"]
            + dnbr_counts_pixels["moderate_high_severity"]
            + dnbr_counts_pixels["moderate_low_severity"]
            + dnbr_counts_pixels["low_severity"]
        )

        st.write("##### Área quemada (en hectáreas):")

        row1_col2_col1, row1_col2_col2 = st.columns([1, 3])
        with row1_col2_col1:
            st.image("img/terreno.png")
        with row1_col2_col2:
            st.metric(
                label="Total", value="{:,.1f}".format(hectareas_quemadas), delta=""
            )

        st.write(f"**Baja severidad**: {dnbr_counts_pixels['low_severity']}")
        st.write(
            f"**Mediana severidad**: {(dnbr_counts_pixels['moderate_high_severity'] + dnbr_counts_pixels['moderate_low_severity'])}"
        )
        st.write(f"**Alta severidad**: {dnbr_counts_pixels['high_severity']}")
