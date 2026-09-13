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


def cargar_rutas():
    """Envoltura SIN cache: comprueba que los archivos existan antes de leer.

    El chequeo va aqui y no dentro de la funcion cacheada a proposito. Si el
    @st.cache_data envolviera tambien el fallo, un None quedaria guardado y la
    app seguiria sin rutas aunque los archivos llegaran despues, hasta que
    alguien la reiniciara. Paso exactamente eso al desplegar."""
    carpeta = Path(__file__).resolve().parent
    if not ((carpeta / "municipios.parquet").exists()
            and (carpeta / "rutas.parquet").exists()):
        return None, None
    return _leer_rutas()


@st.cache_data
def _leer_rutas():
    """Tabla de municipios y distancias, construida por scripts/02 desde el
    propio RNDC. No se usa Google Maps: su API pide tarjeta de crédito y, más
    importante, daría SU distancia y no la que el modelo aprendió."""
    carpeta = Path(__file__).resolve().parent
    mun = pd.read_parquet(carpeta / "municipios.parquet").sort_values("nombre")
    rut = pd.read_parquet(carpeta / "rutas.parquet")
    # cod origen + cod destino -> kilómetros, para buscar en un solo paso
    distancias = {
        (o, d): int(k)
        for o, d, k in zip(rut.codmunicipioorigen, rut.codmunicipiodestino, rut.km)
    }
    return mun, distancias


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
municipios, distancias = cargar_rutas()

# ------------------------------------------------------------
# Ruta (opcional): completa departamentos y distancia
# ------------------------------------------------------------
# Va FUERA del formulario a propósito. Dentro de un st.form los widgets no
# reaccionan hasta que se envía, y aquí se necesita que al escoger los
# municipios se actualicen de inmediato los campos de abajo.
km_ruta = dpto_origen = dpto_destino = None

if municipios is not None:
    with st.container(border=True):
        st.markdown("**¿Sabes los municipios?** Completo el resto.")

        # Las opciones son los CODIGOS de municipio; format_func decide que
        # se lee en pantalla. Asi el nombre es solo presentacion y la llave
        # real sigue siendo el codigo, que es lo que busca la tabla de rutas.
        codigos = municipios["cod"].tolist()
        etiqueta = dict(zip(municipios["cod"], municipios["nombre"]))

        with st.container(horizontal=True):
            m_origen = st.selectbox("Municipio de origen", codigos, index=None,
                                    format_func=etiqueta.get,
                                    placeholder="Escribe para buscar")
            m_destino = st.selectbox("Municipio de destino", codigos, index=None,
                                     format_func=etiqueta.get,
                                     placeholder="Escribe para buscar")

        if m_origen and m_destino:
            fila_o = municipios[municipios.cod == m_origen].iloc[0]
            fila_d = municipios[municipios.cod == m_destino].iloc[0]
            dpto_origen, dpto_destino = fila_o.departamento, fila_d.departamento
            km_ruta = distancias.get((m_origen, m_destino))

            if km_ruta:
                st.success(
                    f"{etiqueta[m_origen]} → {etiqueta[m_destino]}:  "
                    f"{km_ruta:,} km".replace(",", "."),
                    icon=":material/route:",
                )
            else:
                st.info(
                    "Esa ruta no aparece en el RNDC, así que no tengo su distancia. "
                    "Los departamentos sí quedaron; escribe los kilómetros abajo.",
                    icon=":material/info:",
                )

# La clave incluye la ruta: al cambiarla, Streamlit recrea los widgets de
# abajo con los valores nuevos en vez de conservar los que el usuario veía.
clave = f"{m_origen}_{m_destino}" if municipios is not None else "sin_ruta"


def indice(lista, valor, por_defecto=0):
    """Posición de un valor en la lista de categorías, o el de por defecto."""
    return lista.index(valor) if valor in lista else por_defecto


# ------------------------------------------------------------
# Datos del viaje
# ------------------------------------------------------------
# Sin st.form a proposito. Dentro de un formulario los widgets no reaccionan
# hasta enviar, y aqui hacen falta dos cosas inmediatas: que al cambiar la
# unidad se convierta el peso, y que la ruta complete los campos de abajo.
# El calculo igual solo corre al pulsar el boton, asi que no se recalcula de
# mas; el resultado se guarda en session_state para que no desaparezca en el
# siguiente refresco.
st.subheader("Datos del viaje")

with st.container(horizontal=True):
    km = st.number_input(
        "Distancia (km)", min_value=1, max_value=5000, step=10,
        value=int(km_ruta) if km_ruta else 500,
        key=f"km_{clave}", help="Se completa solo si escogiste la ruta arriba",
    )

    # --- Peso, con unidad a elegir ---
    # El valor vive en session_state bajo la clave del widget. Al cambiar de
    # unidad se reescribe ANTES de dibujarlo, asi el numero se convierte en
    # vez de quedarse igual con otro significado.
    if "peso_valor" not in st.session_state:
        st.session_state.peso_valor = 15000.0
        st.session_state.unidad_previa = "kg"

    unidad = st.segmented_control(
        "Unidad del peso", ["kg", "toneladas"], default="kg", key="unidad_peso",
    ) or "kg"

    if unidad != st.session_state.unidad_previa:
        if unidad == "toneladas":
            st.session_state.peso_valor = st.session_state.peso_valor / 1000
        else:
            st.session_state.peso_valor = st.session_state.peso_valor * 1000
        st.session_state.unidad_previa = unidad

    if unidad == "toneladas":
        # Dos decimales, no tres: con %.3f un valor de 15 toneladas se dibuja
        # como 15.000, que en Colombia se lee como quince mil. Justo la
        # confusion que este selector viene a evitar.
        peso = st.number_input("Peso de la carga (toneladas)", min_value=0.1,
                               max_value=60.0, step=0.5, format="%.2f", key="peso_valor")
        kg = peso * 1000
    else:
        peso = st.number_input("Peso de la carga (kg)", min_value=1.0,
                               max_value=60000.0, step=500.0, format="%.0f", key="peso_valor")
        kg = peso


operacion = st.segmented_control(
    ETIQUETAS["operaciontransporte"], CAT["operaciontransporte"],
    # segmented_control recibe el VALOR por defecto, no un indice.
    default=("General" if "General" in CAT["operaciontransporte"]
             else CAT["operaciontransporte"][0]),
)

vehiculo = st.selectbox(ETIQUETAS["config_vehiculo"], CAT["config_vehiculo"])

with st.container(horizontal=True):
    origen = st.selectbox(
        ETIQUETAS["departamentoorigen"], CAT["departamentoorigen"],
        index=indice(CAT["departamentoorigen"], dpto_origen), key=f"do_{clave}",
    )
    destino = st.selectbox(
        ETIQUETAS["departamentodestino"], CAT["departamentodestino"],
        index=indice(CAT["departamentodestino"], dpto_destino), key=f"dd_{clave}",
    )

with st.container(horizontal=True):
    # Arranca en "Carga Normal" (la gran mayoria de los viajes). Sin esto abre
    # en ".", un valor basura del 0,1% de los datos.
    naturaleza = st.selectbox(
        ETIQUETAS["naturalezacarga"], CAT["naturalezacarga"],
        index=indice(CAT["naturalezacarga"], "Carga Normal"),
    )
    mercancia = st.selectbox(ETIQUETAS["categoria_mercancia"], CAT["categoria_mercancia"])

if st.button("Calcular", icon=":material/calculate:", type="primary"):
    if operacion is None:
        st.warning("Escoge un tipo de operación.", icon=":material/warning:")
        st.stop()

    st.session_state.resultado = estimar(paquete, {
        "kilometros": km,
        "kilogramos": kg,
        "config_vehiculo": vehiculo,
        "operaciontransporte": operacion,
        "departamentoorigen": origen,
        "departamentodestino": destino,
        "naturalezacarga": naturaleza,
        "categoria_mercancia": mercancia,
    })
    st.session_state.datos_usados = {"kilometros": km, "kilogramos": kg}


# ------------------------------------------------------------
# Resultado
# ------------------------------------------------------------
# Se lee de session_state y no de la pulsacion del boton: asi sobrevive a los
# refrescos que provoca cualquier otro widget de la pagina.
if "resultado" in st.session_state:
    r = st.session_state.resultado
    usados = st.session_state.datos_usados
    bajo, tipico, alto = r[0.10], r[0.50], r[0.90]

    with st.container(border=True):
        st.metric("Precio típico", pesos(tipico))
        # En Markdown, $...$ delimita una formula. Sin escapar, los dos signos
        # de peso se emparejan y el rango se renderiza como LaTeX, perdiendo
        # los simbolos. st.metric no sufre esto porque no usa Markdown.
        rango = f"{pesos(bajo)} a {pesos(alto)}".replace("$", "\\$")
        st.write(f"**Rango habitual:** {rango}")
        # Los numeros se formatean por separado: aplicar un replace sobre la
        # frase completa tambien cambiaria la puntuacion del texto.
        _km = f"{usados['kilometros']:,.0f}".replace(",", ".")
        _kg = f"{usados['kilogramos']:,.0f}".replace(",", ".")
        _ton = f"{usados['kilogramos'] / 1000:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
        st.caption(
            f"Para {_km} km y {_kg} kg ({_ton} toneladas). "
            "Pedir menos del extremo bajo es quedarse corto frente al mercado; "
            "pedir más del alto es salirse de lo que se pagó en viajes como este."
        )

    fuera = [
        f"{nombre} ({valor:,.0f}) está fuera del rango observado "
        f"({lim[0]:,} a {lim[1]:,})"
        for nombre, valor, lim in [
            ("la distancia", usados["kilometros"], LIMITES["kilometros"]),
            ("el peso", usados["kilogramos"], LIMITES["kilogramos"]),
        ]
        if not (lim[0] <= valor <= lim[1])
    ]
    if fuera:
        st.warning(
            " y ".join(fuera).capitalize() + ". La estimación no es confiable.",
            icon=":material/warning:",
        )

st.divider()

nota_rutas = ""
if municipios is not None:
    nota_rutas = (
        f" Las distancias salen de {len(distancias):,} rutas registradas en el RNDC "
        f"entre {len(municipios):,} municipios; si tu ruta no está, escribe los "
        "kilómetros a mano."
    )

st.caption(
    f"Modelo entrenado con {paquete['n_entrenamiento']:,} viajes de 2015. "
    f"Error medio {paquete['error_medio_pct']}%: el rango contiene el precio real "
    f"el {paquete['cobertura_rango_pct']}% de las veces." + nota_rutas +
    " Los precios **no están ajustados por inflación**, así que sirven para comparar "
    "entre viajes, no como tarifa de hoy."
)
