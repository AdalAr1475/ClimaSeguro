# app/utils/file_loader.py

import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FILE = "clima_huancavelica.csv"

def cargar_csv(nombre_archivo: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el dataset de clima desde un archivo CSV.
    """
    archivo = DATA_DIR / (nombre_archivo or DEFAULT_FILE)

    if not archivo.exists():
        raise FileNotFoundError(f"El archivo {archivo} no fue encontrado.")

    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}")

    return df


def obtener_ultima_lectura(df: pd.DataFrame, distrito: Optional[str] = None) -> dict:
    """
    Devuelve la última lectura climática disponible.
    Si se indica un distrito, filtra solo ese distrito.
    """
    if distrito:
        df = df[df["distrito"].str.upper() == distrito.upper()]

    if df.empty:
        return {}

    df_ordenado = df.sort_values(by=["ano", "semana"], ascending=True)
    ultima = df_ordenado.iloc[-1]

    return {
        "departamento": ultima["departamento"],
        "provincia": ultima["provincia"],
        "distrito": ultima["distrito"],
        "ubigeo": ultima["ubigeo"],
        "ano": int(ultima["ano"]),
        "semana": int(ultima["semana"]),
        "tmean": float(ultima["tmean"]),
        "tmax": float(ultima["tmax"]),
        "tmin": float(ultima["tmin"]),
        "humr": float(ultima["humr"]),
        "ptot": float(ultima["ptot"]),
    }


def filtrar_por_distrito_y_ano(df: pd.DataFrame, distrito: str, ano: int) -> pd.DataFrame:
    """
    Filtra el DataFrame por distrito y año.
    """
    filtrado = df[
        (df["distrito"].str.upper() == distrito.upper()) &
        (df["ano"] == ano)
    ]
    return filtrado.sort_values(by="semana")
