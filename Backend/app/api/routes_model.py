# app/api/routes_model.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/predict")
def get_prediccion_dummy():
    """
    Endpoint de ejemplo que retorna una predicción dummy.
    """
    return {
        "distrito": "Acobamba",
        "semana_predicha": 6,
        "tmean": 12.3,
        "tmax": 17.2,
        "tmin": 8.5,
        "modelo": "RandomForest"
    }
