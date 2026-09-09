"""
TALLER 1 - PIPELINE RNDC (versión comentada para principiantes)

¿Qué es un "pipeline"? Es una cadena de pasos por la que pasan los datos,
en orden: primero se cargan, luego se limpian, luego se analizan.
Cada paso recibe el resultado del anterior.

Este archivo hace exactamente lo mismo que tu versión original, pero:
  - corrige los errores que impedían que corriera,
  - elimina el código repetido usando "funciones" (explicadas más abajo),
  - explica qué hace cada bloque y por qué.

Los pasos son:
  1. Cargar los datos desde internet y convertir los números
  2. Diagnóstico: cuántas filas, qué tipos, cuántos vacíos
  3. Limpieza: quitar registros inservibles y columnas que no aportan
  4. Dividir en carga física y carga líquida
  5. Explorar cada base con las mismas funciones (tablas y gráficos)
"""
#%%
# ============================================================
# IMPORTS: las "cajas de herramientas" que vamos a usar
# ============================================================
# "import X as Y" trae la librería X y le pone el apodo Y para escribir menos.

import pandas as pd            # pandas: maneja tablas (DataFrames). Es el corazón de todo esto.
import matplotlib.pyplot as plt  # matplotlib: dibuja gráficos. plt es la interfaz básica.
import seaborn as sns          # seaborn: gráficos estadísticos más bonitos, construidos sobre matplotlib.
import requests                # requests: pide cosas por internet (páginas, APIs).

# ============================================================
# FUNCIONES Y RUTAS COMPARTIDAS (viven en utils.py, misma carpeta)
# ============================================================
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

from pathlib import Path

# Configuración global (se hace una sola vez, al inicio):
sns.set_theme(style="whitegrid")                        # estilo visual de todos los gráficos
pd.set_option("display.max_columns", None)              # que pandas muestre TODAS las columnas, no "..."
pd.set_option("display.float_format", "{:,.2f}".format)  # números decimales con separador de miles y 2 decimales





# ============================================================
# PASO 1. CARGA Y CONVERSIÓN DE TIPOS
# ============================================================

# Las variables en MAYÚSCULAS son una convención: significa "constante",
# un valor que no cambia durante el programa.
URL = "https://www.datos.gov.co/resource/vn9u-yhwq.json"  # dirección de la API de Datos Abiertos
LIMITE = 1_100_000  # los guiones bajos en números son solo para leerlos mejor: es 1100000

# requests.get(...) hace la petición a internet. "params" son parámetros
# que van en la dirección; "$limit" le dice a la API cuántas filas máximo queremos.
response = requests.get(URL, params={"$limit": LIMITE})
# Si la API respondió con error (por ejemplo, está caída), esta línea
# detiene el programa y muestra el error. Si todo salió bien, no hace nada.
response.raise_for_status()

# response.json() convierte la respuesta (texto JSON) en una lista de diccionarios,
# y pd.DataFrame(...) convierte esa lista en una tabla. Cada diccionario = una fila.
df = pd.DataFrame(response.json())

# Chequeo de seguridad: si la API devolvió exactamente el límite, es muy probable
# que haya MÁS datos que no recibimos. Mejor avisar que trabajar con datos incompletos.
# "raise ValueError(...)" detiene el programa con un mensaje.
if len(df) >= LIMITE:
    raise ValueError(f"La API devolvió {len(df):,} filas: llegaste al límite, hay que paginar con $offset")

#%%
print("Filas cargadas:", len(df))
print(df.head())


# %%
vars_cat = [
    "a_o",
    "cod_config_vehiculo",
    "config_vehiculo",
    "codoperaciontransporte",
    "operaciontransporte",
    "codmunicipioorigen",
    "municipioorigen",
    "departamentoorigen",
    "codmunicipiodestino",
    "municipiodestino",
    "departamentodestino",
    "codmercancia",
    "mercancia",
    "naturalezacarga"
]
df[vars_cat].describe()

# %%
vars_num = [
    "viajestotales",
    "kilogramos",
    "galones",
    "viajesliquidos",
    "viajesvalorcero",
    "kilometros",
    "valorespagados"
]

#%%
    # Conversión a numérico
for col in vars_num:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[vars_num].describe()
# %%
diccionario = pd.DataFrame({
    "variable": df.columns,
    "tipo": df.dtypes.astype(str),
    "nulos": df.isna().sum(),
    "pct_nulos": round(df.isna().mean()*100, 2),
    "valores_unicos": df.nunique()
})

diccionario
#%%
mostrar(df[vars_num].describe().T, "DESCRIPTIVOS ANTES DE FILTRAR")
# %%
df.info()
#%%
# ============================================================
# GUARDAR RESULTADO: para no volver a descargar cada vez
# ============================================================
RUTA_PARQUET = CARPETA_OUTPUT / "rndc_tipificado.parquet"
# GUARDA: si se salto la celda de conversion numerica, las columnas quedan
# como texto y todo lo que sigue en el script 02 se rompe en silencio.
# Mejor detenerse aqui con un mensaje claro que guardar un parquet malo.
sin_convertir = [c for c in vars_num if not pd.api.types.is_numeric_dtype(df[c])]
if sin_convertir:
    raise TypeError(
        f"Estas columnas siguen siendo texto: {sin_convertir}. "
        "Te saltaste la celda de conversion (el bucle con pd.to_numeric). "
        "Corre el script COMPLETO de arriba a abajo antes de guardar."
    )

df.to_parquet(RUTA_PARQUET, index=False)
print(f"Guardado: {RUTA_PARQUET} ({len(df):,} filas)")

# ============================================================
# FIN DEL SCRIPT 1
# ============================================================
# Aquí termina la carga. El tratamiento de nulos (imputar mercancía y
# departamento a partir del código) se hace en el script 02, en un solo
# lugar y con una sola función, para que el diagnóstico y la imputación
# real no puedan contradecirse.
