
# ============================================================
# IMPORTS Y CARGA DEL PARQUET GENERADO EN EL SCRIPT 2
# ============================================================
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# FUNCIONES Y RUTAS COMPARTIDAS (viven en utils.py, misma carpeta)
# ============================================================
# mostrar() y CARPETA_OUTPUT estaban copiados en los tres scripts. Ahora hay
# una sola version, en utils.py. Estas lineas permiten importarla sin importar
# desde donde se ejecute el script.
import sys
from pathlib import Path

try:
    _CARPETA_SCRIPTS = Path(__file__).resolve().parent
except NameError:            # al correr celdas #%% en VS Code no existe __file__
    _CARPETA_SCRIPTS = Path("C:/Users/f/Downloads/ADD 1/TALLER ANÁLISIS DE DATOS - PRECIOS VIAJES/scripts")
if str(_CARPETA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CARPETA_SCRIPTS))

from utils import mostrar, CARPETA_OUTPUT


sns.set_theme(style="whitegrid")

RUTA_PARQUET = CARPETA_OUTPUT / "rndc_modelo.parquet"

df_modelo = pd.read_parquet(RUTA_PARQUET)
print(f"Leído: {len(df_modelo):,} filas, {df_modelo.shape[1]} columnas")
print(df_modelo.dtypes)



# ============================================================
# DIVISIÓN DE LA BASE DE DATOS EN ENTRENAMIENTO Y PRUEBA
# ============================================================

from sklearn.model_selection import train_test_split

# División 80% entrenamiento  y 20% prueba
df_train, df_test = train_test_split(
    df_modelo,
    test_size=0.20,
    random_state=42
)

# Reiniciamos los indices
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Base de datos completa:", len(df_modelo))
print("Base de entrenamiento:", len(df_train))
print("Base de prueba:", len(df_test))

# ------------------------------------------------------------
#Se deja la bd de ENTRENAMIENTO como df: todo el analisis se hace con ella.
#df_test no se toca hasta el final, para evaluar el modelo una sola vez.
# ------------------------------------------------------------

df = df_train.copy()

print("\nDataFrame utilizado para el análisis:", len(df))