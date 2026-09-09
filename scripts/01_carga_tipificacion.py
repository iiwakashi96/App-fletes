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
from pathlib import Path

# Configuración global (se hace una sola vez, al inicio):
sns.set_theme(style="whitegrid")                        # estilo visual de todos los gráficos
pd.set_option("display.max_columns", None)              # que pandas muestre TODAS las columnas, no "..."
pd.set_option("display.float_format", "{:,.2f}".format)  # números decimales con separador de miles y 2 decimales


# ============================================================
# FUNCIÓN AUXILIAR: mostrar tablas
# ============================================================
# Una "función" es un bloque de código con nombre que puedes llamar
# muchas veces. Se define con "def nombre(parámetros):" y se usa
# escribiendo nombre(valores).
#
# ¿Por qué esta función? En Jupyter existe display() para mostrar
# tablas bonitas, pero en un archivo .py normal display() NO existe
# y el programa se cae. Esta función funciona en los dos casos.

def mostrar(tabla, titulo=None):
    """Imprime una tabla de pandas con un título opcional."""
    # "titulo=None" significa que el título es opcional: si no lo pasas, vale None (nada).
    if titulo:                          # si sí me pasaron un título...
        print(f"\n===== {titulo} =====")  # ...lo imprimo. \n es un salto de línea.
    print(tabla.to_string())            # .to_string() convierte la tabla a texto completo, sin recortar filas.


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
mostrar(df[VARIABLES_NUMERICAS].describe().T, "DESCRIPTIVOS ANTES DE FILTRAR")

# %%
df.info()

#%%
# ============================================================
# GUARDAR RESULTADO: para no volver a descargar cada vez
# ============================================================

CARPETA_OUTPUT = Path(r"C:\Users\f\Downloads\ADD 1\TALLER ANÁLISIS DE DATOS - PRECIOS VIAJES\output")
CARPETA_OUTPUT.mkdir(parents=True, exist_ok=True)   # crea la carpeta (y las intermedias) si no existe

RUTA_PARQUET = CARPETA_OUTPUT / "rndc_tipificado.parquet"
df.to_parquet(RUTA_PARQUET, index=False)
print(f"Guardado: {RUTA_PARQUET} ({len(df):,} filas)")

# %%
#%%
# ============================================================
# DIAGNÓSTICO (SOLO INFORMATIVO): ¿cuántos nulos se recuperarían por código?
# ============================================================

def construir_catalogo(df, pares):
    """codigo -> nombre, a partir de filas donde ambos existen. Si hay varios nombres, el más frecuente."""
    partes = [df[[cod, nom]].rename(columns={cod: "codigo", nom: "nombre"}) for cod, nom in pares]
    tabla = pd.concat(partes).dropna()
    return tabla.groupby("codigo")["nombre"].agg(lambda s: s.mode().iloc[0]).to_dict()

cat_municipio = construir_catalogo(df, [("codmunicipioorigen", "municipioorigen"),
                                        ("codmunicipiodestino", "municipiodestino")])
cat_departamento = construir_catalogo(df, [("codmunicipioorigen", "departamentoorigen"),
                                           ("codmunicipiodestino", "departamentodestino")])
cat_mercancia = construir_catalogo(df, [("codmercancia", "mercancia")])

# (columna_codigo, columna_nombre, catálogo) para cada variable con nulos
pares_imputacion = [
    ("codmunicipioorigen",  "municipioorigen",     cat_municipio),
    ("codmunicipiodestino", "municipiodestino",    cat_municipio),
    ("codmunicipioorigen",  "departamentoorigen",  cat_departamento),
    ("codmunicipiodestino", "departamentodestino", cat_departamento),
    ("codmercancia",        "mercancia",           cat_mercancia),
]

resumen = []
sin_rescate = pd.Series(False, index=df.index)   # filas que quedarían con algún nulo aun imputando

for col_cod, col_nom, catalogo in pares_imputacion:
    nulos = df[col_nom].isna()
    recuperable = nulos & df[col_cod].map(catalogo).notna()
    sin_rescate |= nulos & ~recuperable
    resumen.append({
        "variable": col_nom,
        "nulos": nulos.sum(),
        "recuperables": recuperable.sum(),
        "no_recuperables": (nulos & ~recuperable).sum(),
        "pct_recuperado": round(recuperable.sum() / nulos.sum() * 100, 1) if nulos.sum() else 100.0,
    })

mostrar(pd.DataFrame(resumen), "RECUPERACIÓN POSIBLE POR VARIABLE")

filas_con_nulo = df.isna().any(axis=1).sum()
print(f"\nFilas con algún nulo hoy:            {filas_con_nulo:,}")
print(f"Filas que seguirían con nulo tras imputar: {sin_rescate.sum():,}")
print(f"Filas que se salvarían:               {filas_con_nulo - sin_rescate.sum():,}")
# %%
