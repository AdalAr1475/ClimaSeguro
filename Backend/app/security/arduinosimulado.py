import random
import time
import csv
import json # Importar para manejar JSON
from datetime import datetime
import os
import logging # Para integrar el logging

# --- Configuración del Logging---
from log_manager import setup_logger

logger = setup_logger(__name__)


# --- Rutas de archivos ---
# Considera crear una carpeta para los datos generados, ej. 'sensor_data/'
# Asegúrate de que esta ruta sea accesible para el servidor SSH también.
sensor_csv_output_path = "D:/Proyectos2025/ClimaSeguro/Backend/app/security/datos_arduino_simulado.csv"
current_reading_json_path = "D:/Proyectos2025/ClimaSeguro/Backend/app/security/current_sensor_reading.json"

# Asegurarse de que el directorio exista
os.makedirs(os.path.dirname(sensor_csv_output_path), exist_ok=True)


# Inicializar el archivo CSV con los encabezados si no existe o para sobrescribir
# Puedes cambiar 'w' a 'a' si quieres mantener el historial de ejecuciones previas
with open(sensor_csv_output_path, mode='w', newline='') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(['Dia', 'Hora', 'LdrValorAnalog', 'LdrVoltaje', 'LdrResistencia', 'Temperatura', 'Humedad'])
logger.info(f"Archivo CSV '{sensor_csv_output_path}' preparado con encabezados.")

def leer_datos_simulados():
    """
    Simula la lectura de datos de sensores.
    Retorna un diccionario con los datos crudos simulados.
    """
    # Simulación de datos similares a los reales:
    LdrValorAnalog = random.uniform(0.0, 10.0)   # Simula un valor analógico del LDR
    Temperatura = random.uniform(25.0, 27.0)     # Entre 25°C y 27°C
    Humedad = random.randint(40, 46)             # Entre 40% y 46%
    return {
        'LdrValorAnalog': round(LdrValorAnalog, 2),
        'Temperatura': round(Temperatura, 2),
        'Humedad': Humedad
    }

def calcular_RLDR(Vout, Vin, Rfija):
    """Calcula la resistencia del LDR."""
    if Vout == 0: # Evitar división por cero si LdrValorAnalog es 0 (Vout sería 0)
        return float('inf') # Resistencia infinita si no hay salida
    Rldr = Rfija * (Vin / Vout - 1)
    return Rldr

def generate_and_save_sensor_data():
    """
    Genera una lectura completa del sensor, la guarda en CSV y JSON,
    y la retorna en el formato JSON deseado.
    """
    raw_data = leer_datos_simulados()
    
    ldr_analog = raw_data['LdrValorAnalog']
    if ldr_analog == 0:
        ldr_analog = 0.1 # Evitar problemas de división por cero o valores extremos


    # Calcular valores derivados
    ldr_voltaje = (ldr_analog / 1023.0) * 5.0
    ldr_voltaje = round(ldr_voltaje, 4)

    ldr_resistencia = calcular_RLDR(ldr_voltaje, 5.0, 220)
    ldr_resistencia = round(ldr_resistencia, 2)

    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    dia_actual = now.strftime('%Y-%m-%d')
    hora_actual = now.strftime('%H:%M:%S')

    # --- Guardar en CSV (tu formato original) ---
    csv_row = [dia_actual, hora_actual, ldr_analog, ldr_voltaje, ldr_resistencia, raw_data['Temperatura'], raw_data['Humedad']]
    try:
        with open(sensor_csv_output_path, mode='a', newline='') as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(csv_row)
        logger.debug(f"Datos guardados en CSV: {csv_row}")
    except IOError as e:
        logger.error(f"Error al escribir en el archivo CSV: {e}")

    # --- Preparar datos en formato JSON para "emisión" ---
    sensor_data_json = {
        "timestamp": timestamp_str,
        "sensor_id": "sensor_clima_001", # Identificador único para este sensor
        "temperatura_celsius": raw_data['Temperatura'],
        "humedad_porcentaje": raw_data['Humedad'], # Añadimos humedad
        "luz_ldr_analog": ldr_analog,           # Valor LDR analógico
        "luz_ldr_voltaje": ldr_voltaje,         # Voltaje LDR
        "luz_ldr_resistencia": ldr_resistencia  # Resistencia LDR
        # Puedes decidir cuál de los valores de luz es el más relevante para 'luz_lux'
        # o mantenerlos todos si son útiles para el análisis del cliente.
    }

    # --- Guardar la última lectura en un archivo JSON temporal ---
    try:
        with open(current_reading_json_path, 'w') as f:
            json.dump(sensor_data_json, f, indent=4) # indent para formato legible
        logger.info(f"Última lectura guardada en JSON: {current_reading_json_path}")
        logger.debug(json.dumps(sensor_data_json, indent=4))
    except IOError as e:
        logger.error(f"Error al guardar la lectura en JSON: {e}")

    return sensor_data_json

# --- Bucle principal de simulación ---
if __name__ == "__main__":
    logger.info("Iniciando simulación de sensores de clima.")
    while True:
        try:
            generated_data = generate_and_save_sensor_data()
            
            # Imprimir la lectura generada (opcional, para visualización en consola)
            # logger.info(f"Datos generados y disponibles: {json.dumps(generated_data)}") # Esto puede ser muy verboso

            time.sleep(5) # Espera 5 segundos antes de la próxima lectura

        except KeyboardInterrupt:
            logger.info("Simulación de sensores detenida por el usuario.")
            break
        except Exception as e:
            logger.error(f"Error inesperado durante la simulación: {e}", exc_info=True)
            break