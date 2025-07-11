# app/api/routes_data.py

from fastapi import APIRouter, Query, HTTPException, Request # <--- MODIFICADO: Importar Request
from typing import Optional
from pydantic import BaseModel, validator # <--- MODIFICADO: Importar validator para Pydantic
from datetime import datetime
import utils.file_loader as file_loader # <--- MODIFICADO: Importar desde utils.file_loader

from ..security.log_manager import setup_logger
logger = setup_logger(__name__)

# --- ELIMINADA: Ya NO se importa latest_sensor_reading directamente desde app.main ---
# try:
#     from app.main import latest_sensor_reading
# except ImportError:
#     logger.error("No se pudo importar 'latest_sensor_reading' desde app.main...")
#     latest_sensor_reading = None
# --- FIN DE ELIMINACIÓN ---

router = APIRouter()

# --- MODELO SensorData MODIFICADO para coincidir con la respuesta JSON del sensor ---
class SensorData(BaseModel):
    timestamp: datetime
    sensor_id: str
    temperatura_celsius: float
    humedad_porcentaje: int
    luz_ldr_analog: float
    luz_ldr_voltaje: float
    luz_ldr_resistencia: float

    # <--- NUEVO: Validador para parsear el string de timestamp a objeto datetime
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            # Formato de ejemplo: "2025-07-08 20:30:16"
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return value
    # --- FIN DE VALIDADOR ---


# Endpoint para obtener la última lectura de sensor
# <--- MODIFICADO: Añadir 'request: Request' como parámetro para acceder a app.state
# <--- OPCIONAL: response_model para FastAPI generar la doc Swagger correctamente y validar la salida
@router.get("/latest-sensor-reading", response_model=Optional[SensorData])
async def get_latest_sensor_reading_from_api(request: Request): 
    """
    Devuelve la última lectura de temperatura y luz del sensor de ClimaSeguro,
    obtenida por el backend de FastAPI a través de SSH en segundo plano.
    """
    # <--- MODIFICADO: Acceder a la variable de estado a través de request.app.state
    current_sensor_data = request.app.state.latest_sensor_reading 

    if current_sensor_data:
        logger.info(f"Sirviendo la última lectura del sensor desde la memoria de la API.")
        # <--- NUEVO: Convertir el diccionario recibido del sensor a un modelo Pydantic
        # Esto asegura que la respuesta sea siempre validada y coherente.
        try:
            return SensorData(**current_sensor_data)
        except Exception as e:
            logger.error(f"Error al validar datos del sensor desde la memoria: {e}. Datos: {current_sensor_data}", exc_info=True)
            # Devolver un mensaje de error si los datos en memoria están corruptos
            raise HTTPException(status_code=500, detail="Error interno: Datos de sensor en memoria son inválidos.")
    else:
        logger.warning("GET /latest-sensor-reading: No hay datos de sensor disponibles aún. La recolección de datos puede estar iniciándose o fallando.")
        # <--- MODIFICADO: Devolver 200 OK con mensaje en lugar de HTTPException 404
        return {"message": "Datos de sensor no disponibles aún. La recolección de datos está en curso o falló en el último intento. Intente de nuevo en unos segundos."}

# Endpoint existente
@router.get("/latest")
def get_ultima_lectura(distrito: Optional[str] = Query(None, description="Nombre del distrito")):
    """
    Devuelve la última lectura climática disponible.
    Si se proporciona un distrito, devuelve la última lectura solo de ese distrito.
    """
    try:
        df = file_loader.cargar_csv()
        datos = file_loader.obtener_ultima_lectura(df, distrito=distrito)

        if not datos:
            raise HTTPException(status_code=404, detail="No se encontraron registros para el distrito indicado.") # Mejor usar HTTPException
            # return {"mensaje": "No se encontraron registros para el distrito indicado."}

        logger.info(f"Devolviendo última lectura para distrito '{distrito}': {datos}")
        return datos

    except Exception as e:
        logger.error(f"Error al obtener última lectura: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}") # Usar HTTPException

# <-- AÑADIDO: Nuevo Endpoint POST para recibir datos del sensor
@router.post("/receive-sensor-reading")
async def receive_sensor_reading(data: SensorData):
    """
    Recibe las lecturas de temperatura y luz del sensor enviadas por el ERP.
    """
    logger.info(f"API - Datos de sensor recibidos: Temp={data.temperatura}°C, Luz={data.luz} lux, Timestamp={data.timestamp}")
    
    # --- AQUÍ ES DONDE PROCESARÍAS LOS DATOS REALMENTE ---
    # Por ejemplo:
    # 1. Guardar en una base de datos (MongoDB, PostgreSQL, etc.)
    # 2. Enviar a un sistema de streaming de datos (Kafka, RabbitMQ)
    # 3. Disparar un proceso de análisis o alerta
    # 4. Actualizar un cache en memoria

    # Para esta simulación, simplemente confirmamos la recepción y logeamos.
    return {"message": "Datos de sensor recibidos exitosamente por la API", "received_data": data.dict()}

