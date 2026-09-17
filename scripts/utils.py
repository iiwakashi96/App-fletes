"""
utils.py - Funciones y rutas compartidas por los tres scripts del taller.

¿Por que existe este archivo? Antes la funcion mostrar() estaba copiada en
01, 02 y 03. Tres copias del mismo codigo significan que si un dia quieres
cambiarla (por ejemplo, para que muestre solo las primeras 20 filas) tienes
que acordarte de cambiarla en los tres archivos, y basta olvidar uno para
que los scripts empiecen a comportarse distinto entre si.

Aqui vive una sola version. Los otros scripts la importan con:

    from utils import mostrar, CARPETA_OUTPUT

Lo mismo con la ruta de la carpeta output, que tambien estaba escrita tres
veces: si mueves el proyecto de carpeta, ahora solo cambias una linea.
"""

from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================
# Las rutas se deducen de donde esta ESTE archivo, no se escriben a mano.
# utils.py siempre se importa como modulo, asi que __file__ siempre existe.
# El proyecto ya se movio dos veces (de Descargas a Google Drive, y antes se
# renombro la carpeta); con una ruta fija, cada movida rompia todo y ademas
# el mkdir de abajo recreaba en silencio la carpeta vieja, vacia.
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_OUTPUT = CARPETA_PROYECTO / "output"
CARPETA_OUTPUT.mkdir(parents=True, exist_ok=True)   # la crea si no existe

RUTA_TIPIFICADO = CARPETA_OUTPUT / "rndc_tipificado.parquet"   # sale del script 01
RUTA_MODELO     = CARPETA_OUTPUT / "rndc_modelo.parquet"       # sale del script 02


# ============================================================
# FUNCION AUXILIAR: mostrar tablas
# ============================================================
# En Jupyter existe display() para mostrar tablas bonitas, pero en un archivo
# .py normal display() NO existe y el programa se cae. Esta funcion funciona
# en los dos casos.

def mostrar(tabla, titulo=None):
    """Imprime una tabla de pandas con un titulo opcional."""
    # "titulo=None" significa que el titulo es opcional: si no lo pasas, vale None.
    if titulo:                            # si si me pasaron un titulo...
        print(f"\n===== {titulo} =====")  # ...lo imprimo. \n es un salto de linea.
    print(tabla.to_string())              # .to_string() muestra la tabla completa, sin recortar filas.
