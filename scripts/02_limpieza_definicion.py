#%%
# ============================================================
# IMPORTS Y CARGA DEL PARQUET GENERADO EN EL SCRIPT 1
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
except NameError:
    _CARPETA_SCRIPTS = Path("C:/Users/f/Downloads/ADD 1/PIPELINE - PRECIOS VIAJES/scripts")
if str(_CARPETA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CARPETA_SCRIPTS))

from utils import mostrar, CARPETA_OUTPUT


sns.set_theme(style="whitegrid")

RUTA_PARQUET = CARPETA_OUTPUT / "rndc_tipificado.parquet"

df = pd.read_parquet(RUTA_PARQUET)
print(f"Leído: {len(df):,} filas, {df.shape[1]} columnas")

# GUARDA: verifica que el script 01 dejo las columnas numericas convertidas.
NUMERICAS = ["viajestotales", "kilogramos", "galones", "viajesliquidos",
             "viajesvalorcero", "kilometros", "valorespagados"]
texto = [c for c in NUMERICAS if not pd.api.types.is_numeric_dtype(df[c])]
if texto:
    raise TypeError(
        f"El parquet trae estas columnas como texto: {texto}. "
        "El script 01 se guardo sin convertir. Vuelve a correr el 01 COMPLETO."
    )
print(df.dtypes)


#%%
#  0. Eliminar registros duplicados

# Se hace AQUI, con las 21 columnas completas: en este punto dos filas
# iguales lo son en todo (mismo municipio, misma mercancia, mismos km,
# mismo valor), o sea el mismo registro publicado dos veces. Mas abajo se
# botan municipio y mercancia, y desde ahi dos viajes distintos pueden
# verse iguales sin serlo; esos NO se tocan.
# Quita 57.268 filas del millon; la base final baja de 235.949 a 222.046.
# Sirve para que una misma fila no caiga a los dos lados de la particion
# del script 03 (la filtracion baja del 21,1% al 13,9%).

antes = len(df)
df = df.drop_duplicates()
print(f"DUPLICADOS EXACTOS: eliminados {antes - len(df):,}, quedan {len(df):,}")

#%%
  #1. Limpieza: quitar registros inservibles y columnas que no aportan
  #2. Dividir en carga física y carga líquida
  #3. Explorar cada base con las mismas funciones (tablas y gráficos)
# ============================================================
# COLUMNAS QUE EL PIPELINE NECESITA
# ============================================================
# Antes se hacía df.dropna() a secas, que borra una fila si tiene un nulo
# en CUALQUIER columna, incluidas las 13 que se eliminan más abajo. Se
# perdían filas buenas por un nulo en municipiodestino, que ni siquiera
# entra al modelo. Aquí se lista lo que de verdad se usa.

COLS_MODELO = [           # las que sobreviven y entran al modelo
    "config_vehiculo",
    "operaciontransporte",
    "departamentoorigen",
    "departamentodestino",
    "naturalezacarga",
    "kilogramos",
    "kilometros",
    "valorespagados",
]
COLS_FILTROS = ["viajestotales", "galones"]   # se usan para filtrar y luego se botan
COLS_REQUERIDAS = COLS_MODELO + COLS_FILTROS


# ============================================================
# NULOS: imputar por código y eliminar solo lo irrecuperable
# ============================================================
# Una sola función construye todos los catálogos. Antes el script 01 usaba
# .mode() (asumiendo que un código puede tener varios nombres) y el 02 un
# assert de 1:1: dos reglas distintas para el mismo problema. Ahora es una.
# Además el 01 mostraba que se podían recuperar departamentos por código
# pero el 02 los botaba con dropna(); aquí sí se recuperan.

def construir_catalogo(datos, pares):
    """codigo -> nombre, con las filas donde ambos existen.
    Si un código tiene varios nombres, gana el más frecuente (moda)."""
    partes = [datos[[cod, nom]].rename(columns={cod: "codigo", nom: "nombre"})
              for cod, nom in pares]
    tabla = pd.concat(partes).dropna()
    n_conflictos = (tabla.groupby("codigo")["nombre"].nunique() > 1).sum()
    if n_conflictos:
        print(f"  aviso: {n_conflictos} códigos con más de un nombre; se usa el más frecuente")
    return tabla.groupby("codigo")["nombre"].agg(lambda x: x.mode().iloc[0]).to_dict()

cat_mercancia = construir_catalogo(df, [("codmercancia", "mercancia")])
cat_departamento = construir_catalogo(df, [("codmunicipioorigen", "departamentoorigen"),
                                           ("codmunicipiodestino", "departamentodestino")])

# (columna con nulos, columna código, catálogo)
imputaciones = [
    ("mercancia",           "codmercancia",        cat_mercancia),
    ("departamentoorigen",  "codmunicipioorigen",  cat_departamento),
    ("departamentodestino", "codmunicipiodestino", cat_departamento),
]

resumen = []
for col_nom, col_cod, catalogo in imputaciones:
    nulos = df[col_nom].isna()
    df.loc[nulos, col_nom] = df.loc[nulos, col_cod].map(catalogo)
    recuperados = int(nulos.sum() - df[col_nom].isna().sum())
    resumen.append({"variable": col_nom,
                    "nulos_antes": int(nulos.sum()),
                    "recuperados": recuperados,
                    "quedan_nulos": int(df[col_nom].isna().sum())})
mostrar(pd.DataFrame(resumen), "IMPUTACIÓN POR CÓDIGO")

# --- Eliminación de nulos SOLO en las columnas que se usan ---
antes = len(df)
df = df.dropna(subset=COLS_REQUERIDAS)
print(f"\nRegistros eliminados por nulos en columnas requeridas: {antes - len(df):,}")
print(f"Registros finales: {len(df):,}")


#%%
# ============================================================
# LLAVE DE AGRUPACION (columna GRUPO)
# ============================================================
# Va aqui, despues de la imputacion: esta rellena departamentoorigen y
# departamentodestino, que son parte de la llave. Calcularla antes dejaria
# esas filas con una llave basada en un nulo que luego cambia.
#
# QUE ES: un numero que identifica el TIPO de viaje, no la fila. Se repite
# a proposito: todas las filas que el modelo ve iguales comparten el mismo
# GRUPO. En la base final quedan 129.016 grupos para 222.046 filas.
#
# CON QUE SE CONSTRUYE: exactamente las variables que entran al modelo, ni
# mas ni menos. Sin valorespagados, porque esa es la respuesta, no la
# pregunta: dos viajes iguales con precios distintos son la misma pregunta
# y tienen que quedar en el mismo grupo.
#
# PARA QUE SIRVE: en el script 03, GroupShuffleSplit reparte grupos en vez
# de filas, asi ninguna combinacion queda partida entre entrenamiento y
# prueba. Sin esto, el 51,6% de la base de prueba contiene combinaciones
# que el modelo ya vio, y el examen mide memoria en vez de prediccion.
#
# OJO: GRUPO NO es una variable del modelo. Solo la usa el repartidor. Si
# entrara, get_dummies crearia 129.015 columnas y el modelo aprenderia el
# precio de memoria por grupo, sin mirar kilometros ni peso.
#
# SI CAMBIAN LAS VARIABLES DEL MODELO, esta llave cambia sola, porque se
# deriva de COLS_MODELO.

COLS_LLAVE = [c for c in COLS_MODELO if c != "valorespagados"]

df["GRUPO"] = pd.factorize(df[COLS_LLAVE].astype(str).agg("|".join, axis=1))[0]

print(f"LLAVE GRUPO creada con {len(COLS_LLAVE)} columnas: {COLS_LLAVE}")
print(f"  {df.GRUPO.nunique():,} grupos distintos para {len(df):,} filas")


#PRIMER FILTRO APLICADO METODOLOGICAMENTE, SE DEBE APLICAR ANTES DE CUALQUIER ANÁLISIS, YA QUE SON REGISTROS QUE NO TIENEN SENTIDO PARA EL ANÁLISIS DE PRECIOS DE VIAJES.
#%%
filtro = (
    (df["viajestotales"] == 1)
    & (df["valorespagados"] > 10_000)
    & (df["kilometros"] > 10)
)

n_filtrados = filtro.sum()
print(f"TOTAL DE REGISTROS CON viajestotales == 1, valorespagados > 10.000 y kilometros > 10: "
      f"{n_filtrados:,} de {len(df):,} ({n_filtrados / len(df):.1%})")

antes = len(df)          # se reinicia aquí: si no, el log contaría como
                         # eliminadas por el filtro filas que quitó el dropna
df = df[filtro].copy()
print(f"FILTRO METODOLÓGICO: se conservan {len(df):,} de {antes:,} registros "
      f"({len(df) / antes:.1%}); eliminados {antes - len(df):,}")



#SEGUNDO FILTRO APLICADO: SE TRABAJA SÓLO CON REGISTROS DE VIAJES CON MERCANCÍA SÓLIDA. 
# SEGUNDO FILTRO: análisis por kilogramos. Se conserva solo carga con kilos > 0;
# esto excluye carga exclusivamente líquida (galones sin kilos) y registros sin carga.
# %%
mostrar(
    pd.crosstab(df["galones"] > 0, df["kilogramos"] > 0,
                rownames=["galones > 0"], colnames=["kilogramos > 0"]),
    "CRUCE GALONES vs KILOGRAMOS",
)
#%%

antes = len(df)
df = df[df["kilogramos"] > 0].copy()
print(f"FILTRO KILOGRAMOS: se conservan {len(df):,} de {antes:,}; eliminados {antes - len(df):,}")
#%%
# ============================================================
# SELECCIÓN DE VARIABLES PARA EL MODELO (Y = valorespagados)
# ============================================================
COLS_ELIMINAR = [
    # --- Sin variación tras los filtros aplicados ---
    "a_o",               # un solo valor (2015): una constante no puede explicar variación en Y
    "viajestotales",     # constante = 1 tras el filtro de viaje único
    "viajesliquidos",    # constante = 0 tras filtrar por kilogramos > 0
    "viajesvalorcero",   # constante = 0 tras filtrar valorespagados > 10.000
    "galones",           # ≈ 0 en casi todos los registros tras filtrar por kilos; el residuo es ruido

    # --- Redundantes: código y nombre son la misma información. Se conserva el nombre
    #     por interpretabilidad; tener ambos genera colinealidad perfecta ---
    "cod_config_vehiculo",      # duplica config_vehiculo
    "codoperaciontransporte",   # duplica operaciontransporte
    "codmunicipioorigen",       # duplica municipioorigen
    "codmunicipiodestino",      # duplica municipiodestino
    "codmercancia",             # duplica mercancia

    # --- Alta cardinalidad: informativas pero inmanejables en una primera versión.
    #     Se reemplazan por su versión agregada (departamento, naturalezacarga) ---
    "municipioorigen",    # 1.886 niveles → se usa departamentoorigen (32)
    "municipiodestino",   # 2.791 niveles → se usa departamentodestino (33)
    "mercancia",          # 1.223 niveles → se usa naturalezacarga (8)
]

df_modelo = df.drop(columns=COLS_ELIMINAR)
print(f"Variables para el modelo ({df_modelo.shape[1]}): {list(df_modelo.columns)}")
#%%
# ============================================================
# FUNCIONES DE EXPLORACIÓN (adaptadas a una sola base: df_modelo)
# ============================================================

def tabla_atipicos(datos, variables, k=1.5):
    """Atípicos por rango intercuartílico. k=1.5 estándar, k=3 solo extremos."""
    filas = []
    for v in variables:
        q1, q3 = datos[v].quantile([0.25, 0.75])
        iqr = q3 - q1
        li, ls = q1 - k * iqr, q3 + k * iqr
        n_atip = ((datos[v] < li) | (datos[v] > ls)).sum()
        filas.append({
            "Variable": v, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Limite inferior": li, "Limite superior": ls,
            "Cantidad atipicos": n_atip,
            "Porcentaje atipicos": n_atip / len(datos) * 100,
        })
    return pd.DataFrame(filas)


def tabla_cardinalidad(datos):
    """Número de categorías distintas por columna de texto."""
    cols = datos.select_dtypes(include=["object", "category", "string"]).columns
    return (
        pd.DataFrame({"Variable": cols, "Categorias_Unicas": [datos[c].nunique() for c in cols]})
        .sort_values("Categorias_Unicas", ascending=False)
    )


def graficar_relaciones(datos, n_muestra=50_000):
    """Dispersión km vs valor y kg vs valor, más correlación de Spearman."""
    muestra = datos.sample(n=min(n_muestra, len(datos)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=muestra, x="kilometros", y="valorespagados", alpha=0.3, color="b", ax=axes[0])
    axes[0].set(title="Distancia (km) vs. Valor Pagado ($)", xlabel="Kilómetros", ylabel="Valor Pagado ($)")

    sns.scatterplot(data=muestra, x="kilogramos", y="valorespagados", alpha=0.3, color="g", ax=axes[1])
    axes[1].set(title="Peso (kg) vs. Valor Pagado ($)", xlabel="Peso (kg)", ylabel="Valor Pagado ($)")

    plt.tight_layout()
    plt.show()

    corr = muestra[["valorespagados", "kilometros", "kilogramos"]].corr(method="spearman")
    mostrar(corr, "Correlación de Spearman")


def graficar_categoricas(datos):
    """Boxplots para categóricas de pocos niveles; top 15 para las de muchos."""
    # ----- Pocas categorías: boxplots -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, titulo in zip(
        axes,
        ["operaciontransporte", "naturalezacarga", "config_vehiculo"],
        ["Operación de Transporte", "Naturaleza de Carga", "Configuración de Vehículo"],
    ):
        sns.boxplot(data=datos, x=col, y="valorespagados", hue=col, legend=False, palette="Set2", ax=ax)
        ax.set(title=f"Flete por {titulo}", xlabel=titulo, ylabel="Valor Pagado ($)")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    # ----- Muchas categorías: top 15 por frecuencia, con mediana del flete -----
    datos = datos.assign(
        ruta_dpto=datos["departamentoorigen"] + " -> " + datos["departamentodestino"]
    )
    for col, titulo in [("ruta_dpto", "Rutas por departamento (Origen -> Destino)"),
                        ("departamentodestino", "Departamentos destino")]:
        top = (
            datos.groupby(col)["valorespagados"]
            .agg(conteo="count", flete_mediano="median")
            .nlargest(15, "conteo")
            .reset_index()
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top, x="flete_mediano", y=col, hue=col, legend=False, palette="viridis")
        plt.title(f"Top 15 {titulo} más frecuentes: mediana del flete ($)")
        plt.xlabel("Mediana de Valor Pagado ($)")
        plt.ylabel(titulo)
        plt.tight_layout()
        plt.show()


#%%
# ============================================================
#DATOS ATIPICOS A PARTIR DEL ANALISIS DE IQR
# # ============================================================
VARS_NUM = ["kilogramos", "kilometros", "valorespagados"]

mostrar(df_modelo[VARS_NUM].describe().T, "DESCRIPTIVOS")
mostrar(tabla_atipicos(df_modelo, VARS_NUM, k=1.5), "ATÍPICOS 1.5*IQR")
mostrar(tabla_atipicos(df_modelo, VARS_NUM, k=3), "ATÍPICOS 3*IQR")
mostrar(tabla_cardinalidad(df_modelo), "CARDINALIDAD")
print(f"\nColumnas ({df_modelo.shape[1]}): {df_modelo.columns.tolist()}")

graficar_relaciones(df_modelo)
graficar_categoricas(df_modelo)

#%%
# --- Kilogramos fuera de rango físico ---
print("kg < 100:",     (df_modelo["kilogramos"] < 100).sum())
print("kg > 52.000:",  (df_modelo["kilogramos"] > 52_000).sum())

# --- Tarifa por km y por tonelada-km ---
tmp = df_modelo.assign(
    valor_por_km    = df_modelo["valorespagados"] / df_modelo["kilometros"],
    valor_por_tonkm = df_modelo["valorespagados"] / (df_modelo["kilogramos"] / 1000 * df_modelo["kilometros"]),
)
mostrar(tmp[["valor_por_km", "valor_por_tonkm"]].describe(percentiles=[.01, .05, .5, .95, .99]).T,
        "TARIFAS DERIVADAS")
mostrar(tabla_atipicos(tmp, ["valor_por_km", "valor_por_tonkm"], k=3), "ATÍPICOS 3*IQR TARIFAS")
#%%
livianos = df_modelo[df_modelo["kilogramos"] < 100]
mostrar(livianos["kilogramos"].describe(percentiles=[.25, .5, .75, .9]), "kg < 100: distribución")
mostrar(livianos["config_vehiculo"].value_counts().head(10), "kg < 100: vehículos más frecuentes")
mostrar(livianos["kilogramos"].value_counts().head(15), "kg < 100: valores más repetidos")
# %%
df_modelo["config_vehiculo"].value_counts()

#%%
# ============================================================
# FILTROS DE CALIDAD SOBRE df_modelo
# ============================================================

# --- 1. Kilogramos: peso bruto vehicular máximo (PBV) por configuración, según MinTransporte
pbv_max = {
    "Camión Rígido de 2 ejes": 17000,
    "Camión Rígido de 3 ejes": 28000,
    "Tractocamión de 2 ejes Semiremolque de 1 Eje": 27000,
    "Tractocamión de 2 ejes Semiremolque de 2 Ejes": 32000,
    "Tractocamión de 2 ejes Semiremolque de 3 Ejes": 40500,
    "Tractocamión de 3 ejes Semiremolque de 1 Eje": 28000,
    "Tractocamión de 3 ejes Semiremolque de 2 Ejes": 48000,
    "Tractocamión de 3 ejes Semiremolque de 3 Ejes": 52000,
    "Camión Rígido de 2 ejes Remolque de 2 ejes": 31000,
    "Camión Rígido de 2 ejes Remolque de 3 ejes": 47000,
    "Camión Rígido de 3 ejes Remolque de 2 ejes": 44000,
    "Camión Rígido de 3 ejes Remolque de 3 ejes": 48000,
    "Camión Rígido de 2 ejes Remolque Balanceado de 1 eje": 25000,
    "Camión Rígido de 2 ejes Remolque Balanceado de 2 ejes": 32000,
    "Camión Rígido de 2 ejes Remolque Balanceado de 3 ejes": 32000,
    "Camión Rígido de 3 ejes Remolque Balanceado de 1 eje": 33000,
    "Camión Rígido de 3 ejes Remolque Balanceado de 2 ejes": 40000,
    "Camión Rígido de 3 ejes Remolque Balanceado de 3 ejes": 48000,
}
df_modelo["pbv_max_kg"] = df_modelo["config_vehiculo"].map(pbv_max)
assert df_modelo["pbv_max_kg"].notna().all(), "Hay configuraciones sin PBV en el diccionario"

# --- 2. DECISION: se elimina el peso menor a 100 kg, NO se corrige ---
#
# Antes aqui se asumia que un peso entre 2 y 99 era en realidad toneladas mal
# digitadas, y se multiplicaba por 1000. Se revisaron los datos y la hipotesis
# no se sostiene:
#
#   a) Dentro de kg < 100, la correlacion de Spearman entre peso y valor pagado
#      es NEGATIVA (-0.157). En el resto de la base es POSITIVA (+0.631).
#      A mas "peso", menos plata: eso no es un peso.
#   b) Comparando el mismo tipo de vehiculo, los viajes con kg < 100 cuestan
#      MENOS por kilometro que los de kg >= 100 (razones de 0.67 a 1.02).
#      Si fueran camiones cargados a tope costarian mas, no menos.
#   c) La tarifa implicita por tonelada-km, ya corregida x1000, da 136 pesos
#      contra 384 de la base sana: sigue sin cuadrar, es 3 veces mas barata.
#   d) Solo el 58% de esas filas pasaba el tope de PBV al multiplicar, asi que
#      la correccion conservaba peso inventado y botaba el resto sin criterio.
#
# Conclusion: en esas filas el campo de peso no es usable. El precio se ve
# normal, pero no se puede modelar el peso con un peso que no existe.
#
# LIMITACION PARA EL INFORME: la perdida no es aleatoria. La banda kg < 100 se
# concentra en Camion Rigido de 2 ejes (68% de la banda contra 51% de la base),
# asi que se pierden desproporcionadamente camiones pequenos.

antes = len(df_modelo)
n_livianos = (df_modelo["kilogramos"] < 100).sum()
n_sobrepeso = (df_modelo["kilogramos"] > df_modelo["pbv_max_kg"]).sum()

df_modelo = df_modelo.query("100 <= kilogramos <= pbv_max_kg").drop(columns="pbv_max_kg")

print("FILTRO DE PESO:")
print(f"  eliminados por peso menor a 100 kg (campo no usable): {n_livianos:,}")
print(f"  eliminados por superar el PBV del vehiculo:           {n_sobrepeso:,}")
print(f"  quedan {len(df_modelo):,} de {antes:,} ({len(df_modelo)/antes:.1%})")

#%%
# ============================================================
# SEGUNDO FILTRO DE CALIDAD: VALORES ATIPICOS DE valorespagados
# ============================================================
# El primer filtro de calidad fue por PBV (arriba): descarta pesos que el
# camion no puede cargar. Este es el segundo y ultimo: descarta precios que
# no pueden corresponder a un viaje real.
#
# POR QUE $20.000.000:
#
#   1. La distribucion se rompe sola. Percentiles de valorespagados:
#        p99      $  5.500.000
#        p99,9    $ 11.647.306
#        p99,95   $ 21.000.000
#        p99,99   $112.374.089   <-- se multiplica por diez
#        maximo   $700.000.000
#      No es una cola larga: es otro grupo de datos pegado al final.
#
#   2. Argumento fisico. A la tarifa mediana de esta misma base ($3.509 por
#      km), un flete de $20.000.000 implicaria recorrer 5.700 km. El viaje
#      mas largo de toda la base es de 1.883 km y Colombia de punta a punta
#      son unos 1.700 km. Ningun viaje real puede llegar a esa cifra.
#
#   3. Esas filas no son viajes caros, son viajes normales mal digitados:
#      recorren 491 km de mediana (la base, 442) pero cobran $119.962 por km
#      contra $3.509 de la base: 34 veces la tarifa. De las 135, hay 116 que
#      cobran mas de 10 veces lo normal y 35 que recorren menos de 200 km.
#
# SE PROBARON VARIOS UMBRALES (analisis de sensibilidad):
#
#     umbral    elimina      %        media     mediana     desv.est
#     $ 10M         283   0,12    1.645.963   1.250.000    1.287.093
#     $ 12M         226   0,10    1.648.202   1.250.000    1.295.005
#     $ 15M         169   0,07    1.651.115   1.250.000    1.308.418
#     $ 20M         135   0,06    1.653.354   1.250.000    1.321.697   <-- elegido
#     $ 30M          95   0,04    1.657.113   1.250.000    1.353.503
#     $ 50M          58   0,02    1.662.780   1.250.000    1.428.634
#
# Entre $10M y $50M la media se mueve menos de 1% y la mediana no se mueve
# nada. Esa es justamente la defensa del umbral: no depende del numero exacto.
# Se elige $20.000.000 por ser el mas conservador (elimina pocas filas) que
# sigue quedando muy por encima de cualquier flete posible. Ojo con la
# redaccion en el informe: no es "el mejor umbral" en un sentido tecnico,
# porque el analisis muestra que cualquiera del rango sirve igual; es una
# eleccion razonable y verificada.
#
# QUE ARREGLA: la media casi no cambia (de $1.692.217 a $1.653.354, un 2,3%)
# y la mediana no cambia nada. Lo que se corrige es la DESVIACION ESTANDAR,
# que baja de $2.792.275 a $1.321.697, y el maximo, que baja de
# $700.000.000 a $20.000.000.

TECHO_VALOR = 20_000_000

antes = len(df_modelo)
df_modelo = df_modelo[df_modelo["valorespagados"] <= TECHO_VALOR].copy()

print("FILTRO DE VALORES ATIPICOS:")
print(f"  eliminados por valor mayor a ${TECHO_VALOR:,}: {antes - len(df_modelo):,}")
print(f"  quedan {len(df_modelo):,} de {antes:,} ({len(df_modelo)/antes:.2%})")
mostrar(df_modelo["valorespagados"].describe().to_frame(), "valorespagados FINAL")

#%%
# ============================================================
# GUARDAR BASE LISTA PARA EL MODELO
# ============================================================
RUTA_MODELO = CARPETA_OUTPUT / "rndc_modelo.parquet"

# (Ya no existe kg_corregido: se elimino junto con la correccion x1000, asi que
#  no queda ninguna marca del proceso de limpieza como variable del modelo.)

# Las columnas de texto se guardan como "category" con la lista COMPLETA de
# valores de la base final. Asi, cuando el script 03 parta en entrenamiento y
# prueba, las dos mitades cargan las mismas categorias aunque alguna aparezca
# en una sola de ellas (pasa con config_vehiculo: 17 en una, 18 en la otra).
# Parquet conserva este tipo, asi que el 03 ya no tiene que declararlo.
for _col in df_modelo.select_dtypes(include=["object", "str"]).columns:
    df_modelo[_col] = df_modelo[_col].astype("category")
print("Categoricas declaradas:", list(df_modelo.select_dtypes("category").columns))

# GUARDA: si corriste celdas sueltas y te saltaste filtros, aqui llegarian
# cientos de miles de filas de mas. La base depurada ronda las 236.000.
MAX_ESPERADO = 300_000
if len(df_modelo) > MAX_ESPERADO:
    raise ValueError(
        f"df_modelo tiene {len(df_modelo):,} filas, mas de las {MAX_ESPERADO:,} esperadas. "
        "Falto aplicar algun filtro: corre el script COMPLETO de arriba a abajo."
    )
if "pbv_max_kg" in df_modelo.columns:
    raise ValueError("pbv_max_kg sigue presente: no se ejecuto la celda del filtro de peso.")

df_modelo.to_parquet(RUTA_MODELO, index=False)
print(f"Guardado: {RUTA_MODELO} ({len(df_modelo):,} filas, {df_modelo.shape[1]} columnas)")
# %%
df_modelo.info()
