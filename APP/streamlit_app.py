"""Calculadora de fletes - version web (Streamlit).

Misma logica que app_fletes.py, pero para el navegador. Consulta el modelo
entrenado por scripts/03_modelo.py; no entrena nada.

Local:      streamlit run streamlit_app.py
Despliegue: Streamlit Community Cloud, archivo principal APP/streamlit_app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Calculadora de fletes",
    page_icon=":material/local_shipping:",
)

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_fletes.joblib"

ETIQUETAS = {
    "kilometros": "Distancia",
    "kilogramos": "Peso de la carga",
    "config_vehiculo": "Configuración del vehículo",
    "operaciontransporte": "Tipo de operación",
    "departamentoorigen": "Departamento de origen",
    "departamentodestino": "Departamento de destino",
    "naturalezacarga": "Naturaleza de la carga",
    "categoria_mercancia": "Categoría de mercancía",
}

# Rango observado en los datos de entrenamiento: fuera de aquí no es confiable
LIMITES = {"kilometros": (11, 1883), "kilogramos": (100, 51300)}


@st.cache_resource
def cargar_modelo():
    """El modelo pesa 4 MB; se carga una sola vez y queda en memoria."""
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


def estimar(paquete, datos):
    """Devuelve los tres precios (percentil 10, 50 y 90) para un viaje."""
    viaje = pd.DataFrame([datos])
    for c in paquete["vars_cat"]:
        viaje[c] = pd.Categorical(viaje[c], categories=paquete["categorias"][c])

    columnas = paquete["vars_num"] + paquete["vars_cat"]
    return {
        q: float(np.exp(paquete["arboles"][q].predict(viaje[columnas]))[0])
        for q in paquete["cuantiles"]
    }


def pesos(valor):
    return "$" + f"{valor:,.0f}".replace(",", ".")


# ============================================================
# La página
# ============================================================
st.title("Calculadora de fletes")
st.caption(
    "Estimador de referencia del valor pagado por un viaje de carga por carretera, "
    "a partir del RNDC 2015."
)

paquete = cargar_modelo()

if paquete is None:
    st.error(
        f"No se encontró el modelo en `{RUTA_MODELO.name}`. "
        "Corre `scripts/03_modelo.py` completo para generarlo.",
        icon=":material/error:",
    )
    st.stop()

CAT = paquete["categorias"]

# Un formulario agrupa los ocho campos y solo calcula al enviar, en vez de
# recalcular cada vez que el usuario toca un desplegable.
with st.form("viaje"):
    st.subheader("Datos del viaje")

    with st.container(horizontal=True):
        km = st.number_input(
            ETIQUETAS["kilometros"], min_value=1, max_value=5000, value=500, step=10,
            help="Kilómetros recorridos",
        )
        kg = st.number_input(
            ETIQUETAS["kilogramos"], min_value=1, max_value=60000, value=15000, step=500,
            help="Peso de la carga en kilogramos",
        )

    # Cuatro opciones cortas: se muestran todas, sin desplegar.
    operacion = st.segmented_control(
        ETIQUETAS["operaciontransporte"],
        CAT["operaciontransporte"],
        default=CAT["operaciontransporte"][2],
    )

    vehiculo = st.selectbox(ETIQUETAS["config_vehiculo"], CAT["config_vehiculo"])

    with st.container(horizontal=True):
        origen = st.selectbox(ETIQUETAS["departamentoorigen"], CAT["departamentoorigen"])
        destino = st.selectbox(ETIQUETAS["departamentodestino"], CAT["departamentodestino"])

    with st.container(horizontal=True):
        naturaleza = st.selectbox(ETIQUETAS["naturalezacarga"], CAT["naturalezacarga"])
        mercancia = st.selectbox(ETIQUETAS["categoria_mercancia"], CAT["categoria_mercancia"])

    enviar = st.form_submit_button(
        "Calcular", icon=":material/calculate:", type="primary"
    )


if enviar:
    if operacion is None:
        st.warning("Escoge un tipo de operación.", icon=":material/warning:")
        st.stop()

    datos = {
        "kilometros": km,
        "kilogramos": kg,
        "config_vehiculo": vehiculo,
        "operaciontransporte": operacion,
        "departamentoorigen": origen,
        "departamentodestino": destino,
        "naturalezacarga": naturaleza,
        "categoria_mercancia": mercancia,
    }

    r = estimar(paquete, datos)
    bajo, tipico, alto = r[0.10], r[0.50], r[0.90]

    with st.container(border=True):
        st.metric("Precio típico", pesos(tipico))
        # En Markdown, $...$ delimita una formula matematica. Sin escapar, los
        # dos signos de peso se emparejan y el rango se renderiza como LaTeX,
        # perdiendo los simbolos. st.metric no sufre esto porque no usa Markdown.
        rango = f"{pesos(bajo)} a {pesos(alto)}".replace("$", "\\$")
        st.write(f"**Rango habitual:** {rango}")
        st.caption(
            "Pedir menos del extremo bajo es quedarse corto frente al mercado; "
            "pedir más del alto es salirse de lo que se pagó en viajes como este."
        )

    # Aviso si el viaje se sale del rango con el que se entrenó el modelo.
    fuera = [
        f"{ETIQUETAS[c].lower()} ({valor:,.0f}) está fuera del rango observado "
        f"({lim[0]:,} a {lim[1]:,})"
        for c, valor, lim in [("kilometros", km, LIMITES["kilometros"]),
                              ("kilogramos", kg, LIMITES["kilogramos"])]
        if not (lim[0] <= valor <= lim[1])
    ]
    if fuera:
        st.warning(
            "La " + "; la ".join(fuera) + ". La estimación no es confiable.",
            icon=":material/warning:",
        )

st.divider()

st.caption(
    f"Modelo entrenado con {paquete['n_entrenamiento']:,} viajes de 2015. "
    f"Error medio {paquete['error_medio_pct']}%: el rango contiene el precio real "
    f"el {paquete['cobertura_rango_pct']}% de las veces. "
    "Los precios **no están ajustados por inflación**, así que sirven para comparar "
    "entre viajes, no como tarifa de hoy."
)
