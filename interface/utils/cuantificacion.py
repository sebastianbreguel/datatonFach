import rasterio
import rioxarray

# Esta

tif = "C:/Users/lucas/OneDrive/Desktop/Dataton-Fach/rasters/lansat/2023_valpo_swir16-nir-red.tif"
src_img = rasterio.open(tif)

import rasterio
from rasterio.plot import show

# Abre el archivo TIFF
with rasterio.open(tif) as src:
    # Lee la banda que te interesa (por ejemplo, la primera banda)
    banda_nir_red = src.read(1)  # Puedes cambiar el número de banda según tus necesidades

# Muestra los valores de la banda como números
print(banda_nir_red)
