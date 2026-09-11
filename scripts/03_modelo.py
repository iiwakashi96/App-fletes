
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
except NameError:            # al correr celdas #%% no existe __file__
    # Se busca utils.py partiendo de la carpeta de trabajo, sin nombres fijos.
    _CARPETA_SCRIPTS = next(
        (c for c in (Path.cwd() / "scripts", Path.cwd(), Path.cwd().parent / "scripts")
         if (c / "utils.py").exists()),
        Path.cwd(),
    )
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

from sklearn.model_selection import GroupShuffleSplit

# División 80% entrenamiento y 20% prueba, POR GRUPOS.
#
# train_test_split reparte filas sueltas, y eso parte los grupos: la misma
# combinación de viaje puede caer a los dos lados. El modelo estudia la fila
# de entrenamiento y luego le preguntas la de prueba, que es la misma
# pregunta. Medido: el 51,6% de la base de prueba tenía ese problema.
#
# GroupShuffleSplit reparte GRUPOS: baraja los 129.016 valores distintos de
# la columna GRUPO (creada en el script 02) y los va echando al montón de
# prueba hasta acercarse al 20% de las FILAS. Un grupo nunca se corta.
#
# Devuelve posiciones, no tablas, por eso van dos líneas en vez de una:
#   n_splits=1  -> una sola partición (puede generar varias)
#   next(...)   -> .split() devuelve un generador; next() pide el primero
#
# El 20% queda aproximado porque los grupos son piezas indivisibles. Con
# 129.016 grupos el desajuste es despreciable.

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
i_train, i_test = next(gss.split(df_modelo, groups=df_modelo["GRUPO"]))

df_train = df_modelo.iloc[i_train]
df_test = df_modelo.iloc[i_test]

# Reiniciamos los indices
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Base de datos completa:", len(df_modelo))
print("Base de entrenamiento:", len(df_train))
print("Base de prueba:", len(df_test))

# Comprobación: ningún grupo debe estar en los dos lados.
comunes = set(df_train["GRUPO"]) & set(df_test["GRUPO"])
assert not comunes, f"{len(comunes)} grupos quedaron partidos entre entrenamiento y prueba"
print(f"Sin filtración: 0 grupos compartidos "
      f"({df_train.GRUPO.nunique():,} grupos entrenan, {df_test.GRUPO.nunique():,} evalúan)")

# ------------------------------------------------------------
#Se deja la bd de ENTRENAMIENTO como df: todo el analisis se hace con ella.
#df_test no se toca hasta el final, para evaluar el modelo una sola vez.
# ------------------------------------------------------------

df = df_train.copy()

print("\nDataFrame utilizado para el análisis:", len(df))

#%%
# ============================================================
# PREPARACIÓN DE LAS VARIABLES
# ============================================================
import numpy as np

VARS_CAT = ["config_vehiculo", "operaciontransporte", "departamentoorigen",
            "departamentodestino", "naturalezacarga"]
VARS_NUM = ["kilometros", "kilogramos"]
OBJETIVO = "valorespagados"

# GRUPO no aparece en ninguna de las dos listas a propósito: solo sirvió para
# repartir. Si entrara al modelo, get_dummies crearía 129.015 columnas y el
# modelo aprendería el precio de memoria por grupo, ignorando km y peso.

# Las categóricas ya vienen declaradas como "category" desde el script 02, con
# la lista completa de valores. Por eso aquí no hay que convertir nada: al partir
# 80/20, las dos mitades conservan las mismas categorías aunque alguna solo
# aparezca en una de ellas. Esta verificación avisa si el parquet viniera viejo.
sin_declarar = [c for c in VARS_CAT if not isinstance(df[c].dtype, pd.CategoricalDtype)]
if sin_declarar:
    raise TypeError(
        f"Estas columnas no vienen como category: {sin_declarar}. "
        "Vuelve a correr el script 02 COMPLETO para regenerar el parquet."
    )

# ¿Por qué logaritmo? Los fletes van de 10.000 a 20.000.000 de pesos. Trabajando
# en pesos, un error de 2 millones en un viaje enorme le pesa muchísimo más al
# modelo que un error de 50.000 en uno pequeño, así que el modelo se dedica a
# complacer a los viajes grandes. El logaritmo comprime la escala.
y_train = np.log(df[OBJETIVO])
y_test_real = df_test[OBJETIVO].to_numpy(float)   # este queda en pesos, para evaluar


def evaluar(pred_en_pesos, nombre):
    """Compara predicciones (en pesos) contra los precios reales de df_test."""
    err = np.abs(pred_en_pesos - y_test_real) / y_test_real
    print(f"\n----- {nombre} -----")
    print(f"  error medio            : {err.mean()*100:.1f}%")
    print(f"  acierta dentro de +-10%: {(err <= .10).mean()*100:.0f}% de los viajes")
    print(f"  acierta dentro de +-20%: {(err <= .20).mean()*100:.0f}%")
    print(f"  acierta dentro de +-30%: {(err <= .30).mean()*100:.0f}%")
    return err.mean()*100


#%%
# ============================================================
# MODELO 1: REGRESIÓN LINEAL EN LOGARITMO  (para el informe)
# ============================================================
# Este es el modelo que EXPLICA. No es el que mejor predice, pero es el único
# que entrega coeficientes que se pueden leer en una frase.
import statsmodels.api as sm

# get_dummies convierte cada categoría en una columna de 0 y 1.
# drop_first=True elimina una categoría de cada variable: esa se vuelve la
# "categoría de referencia" y las demás se interpretan comparadas contra ella.
X_train_lin = pd.get_dummies(df[VARS_NUM + VARS_CAT], columns=VARS_CAT,
                             drop_first=True, dtype=float)
X_train_lin[VARS_NUM] = np.log(X_train_lin[VARS_NUM])

X_test_lin = pd.get_dummies(df_test[VARS_NUM + VARS_CAT], columns=VARS_CAT,
                            drop_first=True, dtype=float)
X_test_lin[VARS_NUM] = np.log(X_test_lin[VARS_NUM])
X_test_lin = X_test_lin.reindex(columns=X_train_lin.columns, fill_value=0.0)

modelo_lineal = sm.OLS(y_train, sm.add_constant(X_train_lin)).fit()

print("=" * 60)
print("MODELO 1: REGRESIÓN LINEAL EN LOGARITMO")
print("=" * 60)
print(f"  observaciones : {int(modelo_lineal.nobs):,}")
print(f"  variables     : {len(modelo_lineal.params)}")
print(f"  R2 ajustado   : {modelo_lineal.rsquared_adj:.4f}")

# --- LAS ELASTICIDADES: el resultado principal para el informe ---
# Como Y y las dos numéricas están en logaritmo, el coeficiente se lee como
# porcentaje DIRECTAMENTE: si el coeficiente es 0,46, quiere decir que
# subir 1% la variable sube 0,46% el flete. OJO: no se multiplica por 100.
print("\n  ELASTICIDADES (esto es lo que va en el informe):")
for v in VARS_NUM:
    b = modelo_lineal.params[v]
    p = modelo_lineal.pvalues[v]
    print(f"    si {v:<12} sube 1%  ->  el flete sube {b:.2f}%   (p = {p:.3g})")

# Para ver la tabla completa con todos los coeficientes:
# print(modelo_lineal.summary())

# --- Predicción sobre la base de prueba ---
# Al devolver de logaritmo a pesos con exp() se subestima el promedio. La
# corrección de abajo lo compensa.
correccion = np.exp(modelo_lineal.resid.var() / 2)
pred_lineal = np.exp(modelo_lineal.predict(sm.add_constant(X_test_lin))) * correccion
err_lineal = evaluar(pred_lineal, "MODELO 1 sobre la base de prueba")


#%%
# ============================================================
# MODELO 2: ÁRBOL POTENCIADO POR CUANTILES  (para la herramienta)
# ============================================================
# Este es el modelo que PREDICE. No entrega coeficientes, pero acierta mucho más.
# Se entrena tres veces con el MISMO algoritmo, cambiando solo el cuantil:
#   quantile=0.10 -> apunta bajo     (el precio bajo del rango)
#   quantile=0.50 -> apunta al medio (el precio típico)
#   quantile=0.90 -> apunta alto     (el precio alto del rango)
from sklearn.ensemble import HistGradientBoostingRegressor

CUANTILES = [0.10, 0.50, 0.90]
arboles, pred_arbol = {}, {}

for q in CUANTILES:
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=q,
        categorical_features=VARS_CAT,   # las trata como categorías, sin get_dummies
        max_iter=300, random_state=42,
    )
    m.fit(df[VARS_NUM + VARS_CAT], y_train)
    arboles[q] = m
    # Aquí NO hace falta corrección: al devolver de logaritmo a pesos, exp()
    # conserva los cuantiles exactamente (a diferencia del promedio).
    pred_arbol[q] = np.exp(m.predict(df_test[VARS_NUM + VARS_CAT]))
    print(f"  cuantil {q:.2f} entrenado")

print("\n" + "=" * 60)
print("MODELO 2: ÁRBOL POTENCIADO POR CUANTILES")
print("=" * 60)
err_arbol = evaluar(pred_arbol[0.50], "MODELO 2 (precio típico) sobre la base de prueba")

# --- ¿El rango es honesto? ---
# Si el modelo dice "entre X y Y con 80% de confianza", debería contener el
# precio real el 80% de las veces. Esto lo verifica.
dentro = (y_test_real >= pred_arbol[0.10]) & (y_test_real <= pred_arbol[0.90])
print(f"\n  El rango 10-90 contiene el precio real el {dentro.mean()*100:.1f}% de las veces")
print("  (debería ser 80%: si se parece, el modelo mide bien su incertidumbre)")
print(f"  ancho mediano del rango: {np.median(pred_arbol[0.90] - pred_arbol[0.10]):,.0f} pesos")


#%%
# ============================================================
# COMPARACIÓN DE LOS DOS MODELOS
# ============================================================
comparacion = pd.DataFrame({
    "modelo": ["1. Regresion lineal (log)", "2. Arbol por cuantiles"],
    "error_medio_pct": [round(err_lineal, 1), round(err_arbol, 1)],
    "para_que_sirve": ["explicar que mueve el precio", "estimar cuanto cobrar"],
})
mostrar(comparacion, "COMPARACIÓN")
print("\n  No compiten: el 1 va en el informe, el 2 va en la herramienta.")


#%%
# ============================================================
# LA HERRAMIENTA: consultar el rango de un viaje
# ============================================================
def cuanto_cobrar(kilometros, kilogramos, config_vehiculo,
                  operaciontransporte, departamentoorigen,
                  departamentodestino, naturalezacarga):
    """Devuelve el rango de precios sugerido para un viaje."""
    viaje = pd.DataFrame([{
        "kilometros": kilometros, "kilogramos": kilogramos,
        "config_vehiculo": config_vehiculo, "operaciontransporte": operaciontransporte,
        "departamentoorigen": departamentoorigen, "departamentodestino": departamentodestino,
        "naturalezacarga": naturalezacarga,
    }])
    for c in VARS_CAT:
        viaje[c] = viaje[c].astype(df[c].dtype)   # mismas categorías que el entrenamiento

    r = {q: float(np.exp(arboles[q].predict(viaje[VARS_NUM + VARS_CAT]))[0]) for q in CUANTILES}
    print(f"\n  Viaje: {kilometros:,} km | {kilogramos:,} kg | {config_vehiculo}")
    print(f"  Rango sugerido : {r[0.10]:,.0f}  a  {r[0.90]:,.0f} pesos")
    print(f"  Precio tipico  : {r[0.50]:,.0f} pesos")
    return r


# Ejemplo tomado de un viaje real de la base de prueba:
ejemplo = df_test.iloc[0]
cuanto_cobrar(
    kilometros=int(ejemplo.kilometros), kilogramos=int(ejemplo.kilogramos),
    config_vehiculo=ejemplo.config_vehiculo, operaciontransporte=ejemplo.operaciontransporte,
    departamentoorigen=ejemplo.departamentoorigen, departamentodestino=ejemplo.departamentodestino,
    naturalezacarga=ejemplo.naturalezacarga,
)
print(f"  Se pago en realidad: {ejemplo.valorespagados:,.0f} pesos")
