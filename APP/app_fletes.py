"""
CALCULADORA DE FLETES - RNDC 2015
=================================

Aplicacion local de una sola ventana. Pide los datos de un viaje y devuelve
el rango de precios en que se movio el mercado para viajes parecidos.

Usa el modelo entrenado por scripts/03_modelo.py, guardado en
output/modelo_fletes.joblib. No entrena nada: solo consulta.

Para abrirla, doble clic en el archivo, o desde la terminal:
    python app_fletes.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CARGA DEL MODELO
# ============================================================
CARPETA_APP = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA_APP / "modelo_fletes.joblib"

if not RUTA_MODELO.exists():
    raise SystemExit(
        f"No se encontro el modelo en:\n  {RUTA_MODELO}\n\n"
        "Corre primero scripts/03_modelo.py completo para generarlo."
    )

PAQUETE = joblib.load(RUTA_MODELO)
ARBOLES = PAQUETE["arboles"]
CUANTILES = PAQUETE["cuantiles"]
VARS_NUM = PAQUETE["vars_num"]
VARS_CAT = PAQUETE["vars_cat"]
CATEGORIAS = PAQUETE["categorias"]

# Nombres bonitos para las etiquetas de la ventana
ETIQUETAS = {
    "kilometros": "Distancia (km)",
    "kilogramos": "Peso de la carga (kg)",
    "config_vehiculo": "Configuracion del vehiculo",
    "operaciontransporte": "Tipo de operacion",
    "departamentoorigen": "Departamento de origen",
    "departamentodestino": "Departamento de destino",
    "naturalezacarga": "Naturaleza de la carga",
    "categoria_mercancia": "Categoria de mercancia",
}

# Rango con el que se entreno: fuera de aqui la estimacion no es confiable
LIMITES = {"kilometros": (11, 1883), "kilogramos": (100, 51300)}


# ============================================================
# CALCULO
# ============================================================
def estimar(datos):
    """Recibe un diccionario con las 8 variables y devuelve los tres precios."""
    viaje = pd.DataFrame([datos])
    for c in VARS_CAT:
        viaje[c] = pd.Categorical(viaje[c], categories=CATEGORIAS[c])
    for c in VARS_NUM:
        viaje[c] = pd.to_numeric(viaje[c])

    # El modelo predice en logaritmo; exp() lo devuelve a pesos.
    return {q: float(np.exp(ARBOLES[q].predict(viaje[VARS_NUM + VARS_CAT]))[0])
            for q in CUANTILES}


def pesos(valor):
    """Formatea 1234567 como $1.234.567"""
    return "$" + f"{valor:,.0f}".replace(",", ".")


# ============================================================
# VENTANA
# ============================================================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Calculadora de fletes - RNDC 2015")
        self.resizable(False, False)
        self.configure(padx=18, pady=14)

        self.campos = {}
        fila = 0

        tk.Label(self, text="Datos del viaje",
                 font=("Segoe UI", 13, "bold")).grid(row=fila, column=0, columnspan=2,
                                                     sticky="w", pady=(0, 10))
        fila += 1

        # --- Las dos numericas: cajas de texto ---
        for c in VARS_NUM:
            tk.Label(self, text=ETIQUETAS[c], anchor="w").grid(row=fila, column=0,
                                                               sticky="w", pady=3)
            caja = tk.Entry(self, width=34)
            caja.grid(row=fila, column=1, sticky="w", pady=3)
            self.campos[c] = caja
            fila += 1

        # --- Las seis categoricas: menus desplegables ---
        # Se llenan con los valores que el modelo conoce, asi no se puede
        # escribir uno inexistente.
        for c in VARS_CAT:
            tk.Label(self, text=ETIQUETAS[c], anchor="w").grid(row=fila, column=0,
                                                               sticky="w", pady=3)
            menu = ttk.Combobox(self, values=CATEGORIAS[c], width=32, state="readonly")
            menu.grid(row=fila, column=1, sticky="w", pady=3)
            menu.current(0)
            self.campos[c] = menu
            fila += 1

        tk.Button(self, text="Calcular", command=self.calcular,
                  font=("Segoe UI", 10, "bold"), width=16,
                  ).grid(row=fila, column=0, columnspan=2, pady=(14, 8))
        fila += 1

        # --- Zona del resultado ---
        self.marco = tk.LabelFrame(self, text=" Resultado ", padx=12, pady=10)
        self.marco.grid(row=fila, column=0, columnspan=2, sticky="we")
        fila += 1

        self.tipico = tk.Label(self.marco, text="--", font=("Segoe UI", 17, "bold"))
        self.tipico.pack()
        tk.Label(self.marco, text="precio tipico").pack()
        self.rango = tk.Label(self.marco, text="", font=("Segoe UI", 10))
        self.rango.pack(pady=(8, 0))
        self.aviso = tk.Label(self.marco, text="", fg="#b45309", wraplength=330,
                              justify="left")
        self.aviso.pack(pady=(6, 0))

        # --- Pie con las limitaciones ---
        pie = (f"Modelo entrenado con {PAQUETE['n_entrenamiento']:,} viajes de 2015. "
               f"Error medio {PAQUETE['error_medio_pct']}%. El rango contiene el precio "
               f"real el {PAQUETE['cobertura_rango_pct']}% de las veces.\n"
               "Precios de 2015: NO estan ajustados por inflacion.")
        tk.Label(self, text=pie, fg="#555", wraplength=430, justify="left",
                 font=("Segoe UI", 8)).grid(row=fila, column=0, columnspan=2,
                                            sticky="w", pady=(12, 0))

    # --------------------------------------------------------
    def calcular(self):
        datos = {}

        # Las numericas tienen que ser numeros
        for c in VARS_NUM:
            texto = self.campos[c].get().strip().replace(".", "").replace(",", "")
            if not texto:
                messagebox.showwarning("Falta un dato", f"Escribe {ETIQUETAS[c]}.")
                return
            try:
                datos[c] = float(texto)
            except ValueError:
                messagebox.showwarning(
                    "Dato invalido",
                    f"{ETIQUETAS[c]} debe ser un numero.\nRecibi: {texto!r}")
                return
            if datos[c] <= 0:
                messagebox.showwarning("Dato invalido",
                                       f"{ETIQUETAS[c]} debe ser mayor que cero.")
                return

        for c in VARS_CAT:
            datos[c] = self.campos[c].get()

        # Aviso si el viaje se sale del rango con el que se entreno
        fuera = []
        for c, (bajo, alto) in LIMITES.items():
            if not (bajo <= datos[c] <= alto):
                fuera.append(f"{ETIQUETAS[c]} fuera del rango observado ({bajo:,}-{alto:,})")

        try:
            r = estimar(datos)
        except Exception as e:
            messagebox.showerror("Error al calcular", str(e))
            return

        self.tipico.config(text=pesos(r[0.50]))
        self.rango.config(text=f"Rango habitual:  {pesos(r[0.10])}   a   {pesos(r[0.90])}")
        self.aviso.config(
            text="Cuidado: " + "; ".join(fuera) + ". La estimacion no es confiable."
            if fuera else "")


if __name__ == "__main__":
    App().mainloop()
