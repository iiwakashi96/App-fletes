
# ============================================================
# IMPORTS Y CARGA DEL PARQUET GENERADO EN EL SCRIPT 1
# ============================================================
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

CARPETA_OUTPUT = Path(r"C:\Users\f\Downloads\ADD 1\TALLER ANÁLISIS DE DATOS - PRECIOS VIAJES\output")
RUTA_PARQUET = CARPETA_OUTPUT / "rndc_modelo.parquet"

df_modelo = pd.read_parquet(RUTA_PARQUET)
print(f"Leído: {len(df_modelo):,} filas, {df_modelo.shape[1]} columnas")
print(df_modelo.dtypes)

def mostrar(tabla, titulo=None):
    """Imprime una tabla de pandas con un título opcional."""
    # "titulo=None" significa que el título es opcional: si no lo pasas, vale None (nada).
    if titulo:                          # si sí me pasaron un título...
        print(f"\n===== {titulo} =====")  # ...lo imprimo. \n es un salto de línea.
    print(tabla.to_string())            # .to_string() convierte la tabla a texto completo, sin recortar filas.


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
#Se deja la bd de prueba como df para continuar normal todo el analisis con ella. 
# ------------------------------------------------------------

df = df_test.copy()

print("\nDataFrame utilizado para el análisis:", len(df))