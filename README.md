# datatonFach


Integrantes:
- Luis Aros Illanes
- Andres Sebastian de la Fuente
- Lucas Carrasco Estay
- Benjamin Henriquez Soto
- Sebastian Breguel Gonzalez
- Martin Bravo Diaz

# Demo interactiva

## Instalación
0. Instalar [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) para instalar las librerias necesarias.
```bash
pip install poetry
```

1. Clone the repository
```bash
git clone git@github.com:sebastianbreguel/datatonFach.git
```
2. Run `poetry install`
```bash
poetry install
```
## Ejecucción
3. Ejecutar la app de streamlit desde `\interface\`
```bash
cd .\interface\
```
4. Run `streamlit run app.py`
```bash
streamlit run .\app.py
```

# Entrenamiento y predicción
Para usar [DinoV2](https://github.com/facebookresearch/dinov2) con GPU hay que instalar la version de Torch considerando la versión de CUDA y OS, la página oficial de pytorch genera el cmd especifico para la instalacion [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), uego instalar el requirements.txt.

## Instalación
```bash
pip install -r requirements.txt
```
## descarga de datos y modelos
Descargar los datos usados y modelos generados desde (google drive)[https://drive.google.com/file/d/1XkLBKegO08jcm0mbODx3u-xBhpsu4dod/view?usp=sharing]
## Preparar datos de entrenamiento
Ejecutar los scripts desde src ya que los path de los archivos de entrada y salida estan relativos a esta carpeta.
```bash
cd .\src\
```

Se crean las imagenes de entrada con falso RGB usando Elevacion, NDVI y NDVI_std como canales.
```bash
python create_input_images.py
```
Se procesan las imagenes con DinoV2 para obtener los patch embedddings, se crean los archivos **features.npy** y **labels_cls.npy** con los datos para entrenar.
```bash
python create_train_dataset.py
```

## Entrenar
Los modelos entrenados se guardar en la carpeta **.\models**.
```bash
python train_model_from_features.py
```

## Predecir
Las predicciones se guardaran en la carpeta **.\data\predictions** con el timestamp en el nombre del momento de la ejecucción.
```bash
python predict_knn_dinov2.py
```

