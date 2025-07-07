# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_data, routes_model

app = FastAPI(
    title="ClimaSeguro API",
    description="API para exponer datos climáticos históricos y predicciones usando ML",
    version="1.0.0"
)

# Configurar CORS (útil si se conecta desde frontend)
origins = [
    "http://localhost",
    "http://localhost:3000",  # para React, por ejemplo
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas de la API
app.include_router(routes_data.router, prefix="/data", tags=["Datos"])
app.include_router(routes_model.router, prefix="/model", tags=["Modelos"])

# Ruta raíz opcional
@app.get("/")
def read_root():
    return {"mensaje": "Bienvenido a la API de ClimaSeguro"}
