# app/api/routes_data.py

from fastapi import APIRouter, Query
from typing import Optional
from app.utils.file_loader import cargar_csv, obtener_ultima_lectura

router = APIRouter()

@router.get("/latest")
def get_ultima_lectura(distrito: Optional[str] = Query(None, description="Nombre del distrito")):
    """
    Devuelve la última lectura climática disponible.
    Si se proporciona un distrito, devuelve la última lectura solo de ese distrito.
    """
    try:
        df = cargar_csv()
        datos = obtener_ultima_lectura(df, distrito=distrito)

        if not datos:
            return {"mensaje": "No se encontraron registros para el distrito indicado."}

        return datos

    except Exception as e:
        return {"error": str(e)}
