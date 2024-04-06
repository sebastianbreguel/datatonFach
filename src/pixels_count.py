from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# dNBR Range (scaled by 1000)
dnbr_ranges = {
    'enhaced_regrowth_high': (-500, -251),
    'enhaced_regrowth_low': (-250, -101),
    'unburned': (-100, 99),
    'low_severity': (100, 269),
    'moderate_low_severity': (270, 439),
    'moderate_high_severity': (440, 659),
    'high_severity': (660, 1300)
}

def cuantizacion(tif_path):
    """
    Counts the number of pixels in different categories within a given image.

    Parameters:
    tif_path (str): The file path of the TIFF image.

    Returns:
    dict: A dictionary containing the count of pixels in different categories.
    """

    # dNBR Color Map
    dnbr_counts = {
        'enhaced_regrowth_high': 0,
        'enhaced_regrowth_low': 0,
        'unburned': 0,
        'low_severity': 0,
        'moderate_low_severity': 0,
        'moderate_high_severity': 0,
        'high_severity': 0
    }

    # Leer la imagen
    imagen = Image.open(tif_path)

    # Convertir la imagen a un array
    imagen_array = np.array(imagen)

    # Para cada categoria, contar cuantos pixeles estan en el rango
    for category, (min_val, max_val) in dnbr_ranges.items():

        # Pixels in range
        pixels_in_range = np.logical_and(imagen_array >= min_val, imagen_array <= max_val)

        # Count pixels in range
        dnbr_counts[category] = np.sum(pixels_in_range)

    return dnbr_counts
