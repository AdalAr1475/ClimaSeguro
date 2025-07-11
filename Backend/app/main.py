# app/main.py
from contextlib import asynccontextmanager
import asyncio 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_data, routes_model
from app.services.sensor_fetcher import fetch_sensor_data_via_ssh
from .security.log_manager import setup_logger 
from app.security.ssh_server import start_ssh_server_async  # Importar función asíncrona para iniciar el servidor SSH
from app.security.arduinosimulado import start_arduino_simulator_async  # Importar función asíncrona para iniciar el simulador de Arduino

logger = setup_logger(__name__)

# --- Variable global latest_sensor_reading se gestionará ahora en app.state ---
# Ya no es necesario declararla aquí directamente.

# --- Tarea asíncrona en segundo plano para recolectar datos ---
# La tarea ahora recibe la instancia 'app' para acceder a app.state
async def sensor_data_fetch_task(app: FastAPI): # <--- MODIFICADO: Recibe 'app'
    logger.info("Iniciando tarea de recolección de datos del sensor en segundo plano...")
    while True:
        data = await fetch_sensor_data_via_ssh() # Esto funciona, confirmado por tus logs
        if data:
            app.state.latest_sensor_reading = data # <--- MODIFICADO: Actualiza app.state
            logger.info("Última lectura del sensor actualizada en memoria por la tarea de fondo.")
        else:
            logger.warning("La tarea de fondo no pudo obtener datos del sensor. Reintentando...")
        
        # Esperar 10 segundos antes de la siguiente recolección.
        await asyncio.sleep(10) 

# --- Context Manager para el ciclo de vida de la aplicación FastAPI ---
# Esto asegura que la tarea de fondo se inicie al arrancar la API y se detenga al apagarla.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.latest_sensor_reading = None 
    logger.info("Evento de inicio de FastAPI: Iniciando tareas de fondo para ClimaSeguro.")

    # Iniciar el servidor SSH
    ssh_task = asyncio.create_task(start_ssh_server_async())
    logger.info("Servidor SSH iniciado.")

    # Iniciar el simulador de Arduino
    arduino_task = asyncio.create_task(start_arduino_simulator_async())
    logger.info("Simulador de Arduino iniciado.")

    # Inicia la tarea de recolección de datos del sensor
    fetch_task = asyncio.create_task(sensor_data_fetch_task(app)) 

    yield # Aquí se ejecuta la aplicación FastAPI

    # Evento de apagado de la aplicación
    logger.info("Evento de apagado de FastAPI: Cancelando tareas de fondo de ClimaSeguro.")
    fetch_task.cancel()
    ssh_task.cancel()
    arduino_task.cancel()

    try:
        await fetch_task
    except asyncio.CancelledError:
        logger.info("Tarea de recolección de datos del sensor cancelada.")
        raise

    try:
        await ssh_task
    except asyncio.CancelledError:
        logger.info("Servidor SSH cancelado.")
        raise

    try:
        await arduino_task
    except asyncio.CancelledError:
        logger.info("Simulador de Arduino cancelado.")
        raise
# -------------------------------------------------------------------------

app = FastAPI(
    title="ClimaSeguro API",
    description="API para exponer datos climáticos históricos y predicciones usando ML",
    version="1.0.0",
    lifespan=lifespan # <-- IMPORTANTE: Registrar el context manager de ciclo de vida
)

# Configurar CORS (si es necesario para frontends)
origins = [
    "http://localhost",
    "http://localhost:3000",
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
    logger.info("Solicitud GET a la raíz recibida.")
    return {"mensaje": "Bienvenido a la API de ClimaSeguro"}

